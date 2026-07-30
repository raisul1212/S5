from functools import partial
import os
import pickle
import jax
import jax.numpy as np
from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
import optax
from typing import Any, Tuple


def save_checkpoint(state, path, epoch, test_acc, test_loss,
                    args_dict=None, batchnorm=False):
    """Save a checkpoint atomically (write tmp, rename) containing model
    params, optimizer state, BN stats (if applicable), epoch, test_acc,
    test_loss, and a snapshot of args.  Pickle format -- compatible with
    Flax train_state; minimal external dependencies (no orbax pin).

    S5's original train.py saves NOTHING; this helper is added so we
    can reload + extend training (the equivalent of NCB's --init-from).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "epoch": int(epoch),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "params": jax.device_get(state.params),
        "opt_state": jax.device_get(state.opt_state),
        "step": int(state.step),
        "args": args_dict or {},
    }
    if batchnorm and hasattr(state, "batch_stats"):
        ckpt["batch_stats"] = jax.device_get(state.batch_stats)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp, path)


def save_checkpoint_msgpack(state, path, epoch, test_acc, test_loss,
                            args_dict=None, batchnorm=False):
    """Save via Flax's native msgpack serialization.  Designed to round-trip
    correctly with flax.serialization.from_bytes.  The previous pickle
    format's loaded state fails to reproduce training's test_acc at reload
    time; msgpack format is designed for pytree serialization by the Flax
    team, so should preserve everything model.apply needs.

    Layout on disk:
      <path>.msgpack  -- flax.serialization.to_bytes({"params": ..., "batch_stats": ...})
      <path>.meta.pkl -- {"epoch", "test_acc", "test_loss", "step", "args"}
    """
    import flax.serialization as fs
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"params": state.params}
    if batchnorm and hasattr(state, "batch_stats"):
        payload["batch_stats"] = state.batch_stats
    msgpack_bytes = fs.to_bytes(payload)
    meta = {
        "epoch": int(epoch),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "step": int(state.step),
        "args": args_dict or {},
    }
    tmp_mp = path + ".msgpack.tmp"
    tmp_meta = path + ".meta.pkl.tmp"
    with open(tmp_mp, "wb") as f:
        f.write(msgpack_bytes)
    with open(tmp_meta, "wb") as f:
        pickle.dump(meta, f)
    os.replace(tmp_mp, path + ".msgpack")
    os.replace(tmp_meta, path + ".meta.pkl")


def load_checkpoint_msgpack(path, template_state):
    """Round-trip counterpart to save_checkpoint_msgpack.  Returns
    updated state (with loaded params + batch_stats) and meta dict."""
    import flax.serialization as fs
    with open(path + ".msgpack", "rb") as f:
        msgpack_bytes = f.read()
    template_payload = {"params": template_state.params}
    if hasattr(template_state, "batch_stats"):
        template_payload["batch_stats"] = template_state.batch_stats
    restored = fs.from_bytes(template_payload, msgpack_bytes)
    if "batch_stats" in restored:
        new_state = template_state.replace(
            params=restored["params"],
            batch_stats=restored["batch_stats"])
    else:
        new_state = template_state.replace(params=restored["params"])
    with open(path + ".meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return new_state, meta


# LR schedulers
def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = np.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if ssm_lr < lr_min:
        ssm_lr = lr_min

    return lr, ssm_lr, count, opt_acc


def constant_lr(step, base_lr, end_step,  lr_min=None):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # Update state
    state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'] = np.array(lr_val, dtype=np.float32)
    state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val, dtype=np.float32)
    if opt_config in ["BandCdecay"]:
        # In this case we are applying the ssm learning rate to B, even though
        # we are also using weight decay on B
        state.opt_state.inner_states['none'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val, dtype=np.float32)

    return state, step


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(model_cls,
                       rng,
                       padded,
                       retrieval,
                       in_dim=1,
                       bsz=128,
                       seq_len=784,
                       weight_decay=0.01,
                       batchnorm=False,
                       opt_config="standard",
                       ssm_lr=1e-3,
                       lr=1e-3,
                       dt_global=False
                       ):
    """
    Initializes the training state using optax

    :param model_cls:
    :param rng:
    :param padded:
    :param retrieval:
    :param in_dim:
    :param bsz:
    :param seq_len:
    :param weight_decay:
    :param batchnorm:
    :param opt_config:
    :param ssm_lr:
    :param lr:
    :param dt_global:
    :return:
    """

    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (np.ones((2*bsz, seq_len, in_dim)), np.ones(2*bsz))
            integration_timesteps = np.ones((2*bsz, seq_len,))
        else:
            dummy_input = (np.ones((bsz, seq_len, in_dim)), np.ones(bsz))
            integration_timesteps = np.ones((bsz, seq_len,))
    else:
        dummy_input = np.ones((bsz, seq_len, in_dim))
        integration_timesteps = np.ones((bsz, seq_len, ))

    model = model_cls(training=True)
    init_rng, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    # Include "noise" rng at init even when training doesn't use it.
    # Flax records rng collections at init tracing time; if a rng name
    # is not seen at init, apply() will REJECT it later.  The chip-eval
    # sweep applies with rngs={"noise": ...}, so "noise" must be
    # registered here even for vanilla training.  Cost: one extra split
    # per training start.  No effect on training behavior (SSM's
    # fast path at sigma=0 doesn't consume the rng).
    variables = model.init({"params": init_rng,
                            "dropout": dropout_rng,
                            "noise": noise_rng},
                           dummy_input, integration_timesteps,
                           )
    if batchnorm:
        # Flax >= 0.8 returns plain dict, not FrozenDict; .unfreeze() only
        # exists on the legacy FrozenDict.  Use flax.core.unfreeze() which
        # accepts either and returns a plain dict either way.
        from flax.core import unfreeze as _unfreeze
        params = _unfreeze(variables["params"])
        batch_stats = variables["batch_stats"]
    else:
        # Flax >= 0.8 returns plain dict, not FrozenDict; .unfreeze() only
        # exists on the legacy FrozenDict.  Use flax.core.unfreeze() which
        # accepts either and returns a plain dict either way.
        from flax.core import unfreeze as _unfreeze
        params = _unfreeze(variables["params"])
        # Note: `unfreeze()` is for using Optax.

    if opt_config in ["standard"]:
        """This option applies weight decay to C, but B is kept with the
            SSM parameters with no weight decay.
        """
        print("configuring standard optimization setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
           ssm learning rate to B.
        """
        print("configuring optimization with B in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in ["B"] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in ["B"] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=ssm_lr,
                                                              weight_decay=weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["BfastandCdecay"]:
        """This option applies weight decay to both C and B. Note here we apply
           faster global learning rate to B also.

           Extended to recognize MambinoSSM's predictor branch parameters:
             - Lambda_s_re, Lambda_s_im, log_step_s -> 'ssm' group (no wd, ssm_lr)
                                                       same as Lambda_re/im/log_step
             - B_s -> 'regular' group (with wd, faster lr)
                      same as B for the main scan
             - C_s -> 'regular' group (with wd, faster lr)
                      same as C for the main scan
             - W_eps -> 'regular' group (with wd, faster lr)
                        new additive PC matrix; treat as a normal projection
        """
        print("configuring optimization with B in AdamW setup with lr")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm",
                         "Lambda_s_re", "Lambda_s_im", "gate_kappa", "gate_bias"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm",
                         "Lambda_s_re", "Lambda_s_im", "log_step_s", "gate_kappa", "gate_bias"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included 
            with the SSM parameters and uses ssm learning rate.
         """
        print("configuring optimization with C not in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")

    if batchnorm:
        class TrainState(train_state.TrainState):
            batch_stats: Any
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
    else:
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# Train and eval steps
@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label


def prep_batch(batch: tuple,
               seq_len: int,
               in_dim: int) -> Tuple[np.ndarray, np.ndarray, np.array]:
    """
    Take a batch and convert it to a standard x/y format.
    :param batch:       (x, y, aux_data) as returned from dataloader.
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Err... not sure what I should do... Unhandled data type. ")

    # Convert to JAX.
    inputs = np.asarray(inputs.numpy())

    # Grab lengths from aux if it is there.
    lengths = aux_data.get('lengths', None)

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        # Assuming vocab padding value is zero
        inputs = np.pad(inputs, ((0, 0), (0, num_pad)), 'constant', constant_values=(0,))

    # Inputs is either [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = one_hot(np.asarray(inputs), in_dim)

    # If there are lengths, bundle them up.
    if lengths is not None:
        lengths = np.asarray(lengths.numpy())
        full_inputs = (inputs.astype(float), lengths.astype(float))
    else:
        full_inputs = inputs.astype(float)

    # Convert and apply.
    targets = np.array(targets.numpy())

    # If there is an aux channel containing the integration times, then add that.
    if 'timesteps' in aux_data.keys():
        integration_timesteps = np.diff(np.asarray(aux_data['timesteps'].numpy()))
    else:
        integration_timesteps = np.ones((len(inputs), seq_len))

    return full_inputs, targets.astype(float), integration_timesteps


def train_epoch(state, rng, model, trainloader, seq_len, in_dim, batchnorm, lr_params, lambda_pc=0.0):
    """
    Training function for an epoch that loops over batches.

    `lambda_pc` scales MambinoSSM's intrinsic predictive-coding loss.
    Default 0.0 -> harmless for vanilla S5.

    Returns (state, mean_total_loss, step, epoch_metrics) where
    epoch_metrics is a dict:
      { "task_loss": float, "intrinsic_loss": float,
        "per_block_L_int": {"layers_0": float, ..., "layers_{L-1}": float} }
    Empty dict for vanilla S5 (no sows).
    """
    # Store Metrics
    model = model(training=True)
    batch_losses = []
    batch_task_losses = []
    batch_intrinsic_totals = []
    per_block_accum = {}  # {layer_key: [values across batches]}

    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    for batch_idx, batch in enumerate(tqdm(trainloader)):
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        rng, drop_rng = jax.random.split(rng)
        state, loss, task_loss, intrinsic_total, per_block = train_step(
            state,
            drop_rng,
            inputs,
            labels,
            integration_times,
            model,
            batchnorm,
            float(lambda_pc),
        )
        batch_losses.append(loss)
        batch_task_losses.append(task_loss)
        batch_intrinsic_totals.append(intrinsic_total)
        for k, v in per_block.items():
            per_block_accum.setdefault(k, []).append(v)
        lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)

    per_block_mean = {
        k: float(np.mean(np.array(v)))
        for k, v in per_block_accum.items()
    }
    epoch_metrics = {
        "task_loss": float(np.mean(np.array(batch_task_losses))),
        "intrinsic_loss": float(np.mean(np.array(batch_intrinsic_totals))),
        "per_block_L_int": per_block_mean,
    }
    # Return average loss over batches
    return state, np.mean(np.array(batch_losses)), step, epoch_metrics


def validate(state, model, testloader, seq_len, in_dim, batchnorm, step_rescale=1.0, noise_rng_seed=None):
    """Validation function that loops over batches.

    noise_rng_seed: if not None, passes a per-batch 'noise' RNG derived
    from this seed to the model.  Required when the SSM has
    noise_sigma > 0 (chip analysis eval).  When None, no rngs are
    passed (matches training's original eval path exactly).
    """
    model = model(training=False, step_rescale=step_rescale)
    losses, accuracies, preds = np.array([]), np.array([]), np.array([])
    noise_key = jax.random.PRNGKey(noise_rng_seed) if noise_rng_seed is not None else None
    for batch_idx, batch in enumerate(tqdm(testloader)):
        inputs, labels, integration_timesteps = prep_batch(batch, seq_len, in_dim)
        if noise_key is not None:
            noise_key, sub_key = jax.random.split(noise_key)
        else:
            sub_key = None
        loss, acc, pred = eval_step(inputs, labels, integration_timesteps,
                                    state, model, batchnorm, sub_key)
        losses = np.append(losses, loss)
        accuracies = np.append(accuracies, acc)

    aveloss, aveaccu = np.mean(losses), np.mean(accuracies)
    return aveloss, aveaccu


def _sum_intrinsic_losses(intermediates):
    """Recursively sum all values stored under 'intrinsic_loss' keys in
    Flax's intermediates pytree.  MambinoSSM sows one scalar per block
    via `self.sow('intermediates', 'intrinsic_loss', ...)`; for a
    StackedEncoderModel with L blocks this yields L scalars.  We sum
    them so the resulting term `lambda_pc * total_intrinsic` has the
    same shape as Mambino's native loss: task_CE + lambda_pc * sum_l L_int_l.

    For vanilla S5SSM (no sow call), the intermediates dict has no
    'intrinsic_loss' entries and this returns 0.0 -> harmless no-op.
    """
    total = 0.0
    if not isinstance(intermediates, dict):
        return total
    for key, val in intermediates.items():
        if key == "intrinsic_loss":
            # val is a tuple of sown values (Flax stores as tuples)
            if isinstance(val, tuple):
                for v in val:
                    total = total + np.mean(v)
            else:
                total = total + np.mean(val)
        elif isinstance(val, dict):
            total = total + _sum_intrinsic_losses(val)
    return total


def _collect_intrinsic_per_block(intermediates):
    """Walk intermediates pytree; key each `intrinsic_loss` scalar by its
    containing block ID (path segment matching r"layers_\\d+").

    Returns dict {"layers_0": scalar_jnp, ..., "layers_{L-1}": scalar_jnp}
    with STATIC keys (fixed across jit traces because model layer count
    is fixed).  Empty dict for vanilla S5 (no sows) -> compiled-through
    empty pytree, downstream sum() returns Python 0.
    """
    result = {}

    def _walk(node, current_layer=None):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            if k == "intrinsic_loss":
                if current_layer is not None:
                    val = v[0] if isinstance(v, tuple) else v
                    result[current_layer] = np.mean(val)
            else:
                next_layer = k if k.startswith("layers_") else current_layer
                _walk(v, current_layer=next_layer)

    _walk(intermediates)
    return result


def compute_predictor_frobenius(params):
    """Read state.params, compute Frobenius norms of Mambino's predictor
    branch matrices (B_s, C_s, W_eps) + main-branch reference norms (B, C1/C2)
    + |Lambda_s|/|Lambda| magnitude per block.

    Returns dict keyed by block:
      {"layers_0": {"B_s": float, "C_s": float, "W_eps": float,
                    "Lambda_s_abs": float,
                    "B": float, "C1": float, "C2": float,
                    "Lambda_abs": float,
                    "W_eps_over_B": float},
       ...}

    W_eps_over_B is the ratio ||W_eps||_F / ||B||_F — the primary
    "how much is the PC pathway carrying vs the direct input" metric.
    If W_eps_over_B stays near 0 across epochs, the predictor branch
    is inert.  If it grows meaningfully, the PC path is being used.

    For vanilla S5 (no predictor params), returned dict just has main
    B/C1/C2/Lambda entries — safe no-op-ish.
    """
    result = {}

    def _walk(node, current_layer=None):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            if k.startswith("layers_"):
                _walk(v, current_layer=k)
                continue
            if isinstance(v, dict):
                _walk(v, current_layer=current_layer)
                continue
            if current_layer is None:
                continue
            # v is a leaf param array
            if k in ("B", "C1", "C2", "C", "B_s", "C_s", "W_eps",
                     "Lambda_re", "Lambda_im",
                     "Lambda_s_re", "Lambda_s_im"):
                arr = jax.device_get(v)
                import numpy as onp
                frob = float(onp.sqrt(onp.sum(onp.asarray(arr) ** 2)))
                result.setdefault(current_layer, {})[k] = frob

    _walk(params)

    # Post-process: combine Lambda_re/Lambda_im -> |Lambda|,
    # Lambda_s_re/Lambda_s_im -> |Lambda_s|, and compute ratio.
    import numpy as onp
    for lk, sub in result.items():
        lre = sub.pop("Lambda_re", 0.0)
        lim = sub.pop("Lambda_im", 0.0)
        sub["Lambda_abs"] = float(onp.sqrt(lre * lre + lim * lim))
        lsre = sub.pop("Lambda_s_re", 0.0)
        lsim = sub.pop("Lambda_s_im", 0.0)
        sub["Lambda_s_abs"] = float(onp.sqrt(lsre * lsre + lsim * lsim))
        # W_eps over B ratio (primary "PC path activity" metric)
        b_norm = sub.get("B", 0.0)
        w_norm = sub.get("W_eps", 0.0)
        sub["W_eps_over_B"] = float(w_norm / b_norm) if b_norm > 0 else 0.0
    return result


@partial(jax.jit, static_argnums=(5, 6, 7))
def train_step(state,
               rng,
               batch_inputs,
               batch_labels,
               batch_integration_timesteps,
               model,
               batchnorm,
               lambda_pc,
               ):
    """Performs a single training step given a batch of data.

    `lambda_pc` (static arg) scales the predictive-coding intrinsic loss
    aggregated from MambinoSSM's self.sow('intermediates', 'intrinsic_loss', ...)
    calls.  When lambda_pc == 0.0 (default / S5 baseline / Mambino-hero
    setting), the intrinsic loss is computed but contributes 0 to the
    total loss -- it is logged only.  Vanilla S5SSM never calls sow
    so the intermediates dict has no intrinsic_loss entries either way.

    Returns:
        state:           updated TrainState
        loss:            total loss = task_loss + lambda_pc * intrinsic_total
        task_loss:       pure CE loss (for telemetry, no gradient consumer)
        intrinsic_total: sum over blocks of mean(||eps||^2) (0 for vanilla S5)
        per_block:       dict {"layers_i": scalar_L_int} (empty for vanilla S5)
    """
    def loss_fn(params):

        if batchnorm:
            logits, mod_vars = model.apply(
                {"params": params, "batch_stats": state.batch_stats},
                batch_inputs, batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
            )
        else:
            logits, mod_vars = model.apply(
                {"params": params},
                batch_inputs, batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates"],
            )

        task_loss = np.mean(cross_entropy_loss(logits, batch_labels))

        # MambinoSSM's intrinsic loss: per-block breakdown for telemetry,
        # summed for the gradient consumer.  Empty dict for vanilla S5.
        per_block = _collect_intrinsic_per_block(
            mod_vars.get("intermediates", {}))
        if per_block:
            intrinsic_total = sum(per_block.values())
        else:
            intrinsic_total = np.float32(0.0)
        total_loss = task_loss + lambda_pc * intrinsic_total

        return total_loss, (mod_vars, logits, task_loss, intrinsic_total, per_block)

    (loss, (mod_vars, logits, task_loss, intrinsic_total, per_block)), grads = \
        jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss, task_loss, intrinsic_total, per_block


@partial(jax.jit, static_argnums=(4, 5))
def eval_step(batch_inputs,
              batch_labels,
              batch_integration_timesteps,
              state,
              model,
              batchnorm,
              noise_rng=None,
              ):
    """Eval one batch.  If noise_rng is not None, passes it as the
    'noise' rng to model.apply (required when the SSM module has
    noise_sigma > 0 for chip analysis).  Otherwise omits rngs entirely
    to match training's original eval path.
    """
    rngs = {"noise": noise_rng} if noise_rng is not None else None
    apply_kwargs = {"rngs": rngs} if rngs is not None else {}
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                             batch_inputs, batch_integration_timesteps,
                             **apply_kwargs)
    else:
        logits = model.apply({"params": state.params},
                             batch_inputs, batch_integration_timesteps,
                             **apply_kwargs)

    losses = cross_entropy_loss(logits, batch_labels)
    accs = compute_accuracy(logits, batch_labels)

    return losses, accs, logits


# ============================================================================
# Char-LM path (ADDITIVE). The classification train_step / train_epoch /
# validate / eval_step / cross_entropy_loss above are UNTOUCHED -- task="lm"
# selects these instead at the top level (a task-level kill-switch).
# ============================================================================
def lm_cross_entropy(log_probs, targets):
    """Mean per-token NLL in NATS. log_probs: (B,L,V) log-softmax; targets: (B,L) int."""
    tgt = one_hot(targets, log_probs.shape[-1])       # (B,L,V)
    nll = -np.sum(tgt * log_probs, axis=-1)           # (B,L)
    return np.mean(nll)


def prep_lm_batch(batch, seq_len, in_dim):
    """(input_ids, target_ids) int64 tensors -> (one-hot inputs (B,L,in_dim) float,
    target ids (B,L) int, dummy integration_timesteps)."""
    x, y = batch
    x = np.asarray(x.numpy())
    y = np.asarray(y.numpy())
    x = one_hot(x, in_dim)                             # (B,L,in_dim)
    its = np.ones((x.shape[0], x.shape[1]))            # unused by the SSM; call-parity only
    return x, y, its


@partial(jax.jit, static_argnums=(5, 6, 7))
def lm_train_step(state, rng, batch_inputs, batch_targets, batch_its,
                  model, batchnorm, lambda_pc):
    """Mirror of train_step but with the per-position LM loss. Mambino's intrinsic
    loss (sown per block) is still aggregated (lambda_pc default 0 -> logged only)."""
    def loss_fn(params):
        variables = {"params": params}
        mut = ["intermediates"]
        if batchnorm:
            variables["batch_stats"] = state.batch_stats
            mut = ["intermediates", "batch_stats"]
        log_probs, mod_vars = model.apply(
            variables, batch_inputs, batch_its,
            rngs={"dropout": rng}, mutable=mut)
        task_loss = lm_cross_entropy(log_probs, batch_targets)
        per_block = _collect_intrinsic_per_block(mod_vars.get("intermediates", {}))
        intrinsic_total = sum(per_block.values()) if per_block else np.float32(0.0)
        total = task_loss + lambda_pc * intrinsic_total
        return total, (mod_vars, task_loss, intrinsic_total, per_block)

    (loss, (mod_vars, task_loss, intrinsic_total, per_block)), grads = \
        jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss, task_loss, intrinsic_total, per_block


def lm_train_epoch(state, rng, model, trainloader, seq_len, in_dim, batchnorm,
                   lr_params, lambda_pc=0.0, max_steps=0):
    """Mirror of train_epoch for the LM path. Returns (state, mean_loss, step, metrics).
    max_steps>0 caps the number of train steps this 'epoch' (char-LM trains by steps,
    not full passes over 90M bytes; also used for smoke tests)."""
    model = model(training=True)
    batch_losses, batch_task, batch_intr = [], [], []
    per_block_accum = {}
    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params
    for batch_idx, batch in enumerate(tqdm(trainloader)):
        if max_steps and batch_idx >= max_steps:
            break
        inputs, targets, its = prep_lm_batch(batch, seq_len, in_dim)
        rng, drop_rng = jax.random.split(rng)
        state, loss, task_loss, intr, per_block = lm_train_step(
            state, drop_rng, inputs, targets, its, model, batchnorm, float(lambda_pc))
        batch_losses.append(loss)
        batch_task.append(task_loss)
        batch_intr.append(intr)
        for k, v in per_block.items():
            per_block_accum.setdefault(k, []).append(v)
        lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)
    metrics = {
        "task_loss": float(np.mean(np.array(batch_task))),
        "intrinsic_loss": float(np.mean(np.array(batch_intr))),
        "per_block_L_int": {k: float(np.mean(np.array(v))) for k, v in per_block_accum.items()},
    }
    return state, np.mean(np.array(batch_losses)), step, metrics


@partial(jax.jit, static_argnums=(3, 4))
def lm_eval_step(batch_inputs, batch_its, state, model, batchnorm):
    if batchnorm:
        return model.apply({"params": state.params, "batch_stats": state.batch_stats},
                           batch_inputs, batch_its)
    return model.apply({"params": state.params}, batch_inputs, batch_its)


def lm_validate(state, model, testloader, seq_len, in_dim, batchnorm, max_batches=0):
    """Bits-per-character (BPC) = mean per-token NLL (nats) / ln 2 over the split.
    max_batches>0 caps eval batches (smoke tests); 0 = full split."""
    model = model(training=False)
    tot_nll, tot_tok = 0.0, 0
    for bi, batch in enumerate(tqdm(testloader)):
        if max_batches and bi >= max_batches:
            break
        inputs, targets, its = prep_lm_batch(batch, seq_len, in_dim)
        log_probs = lm_eval_step(inputs, its, state, model, batchnorm)
        tgt = one_hot(targets, log_probs.shape[-1])
        nll = -np.sum(tgt * log_probs, axis=-1)       # (B,L) nats
        tot_nll += float(np.sum(nll))
        tot_tok += int(nll.size)
    return (tot_nll / tot_tok) / float(np.log(2))
