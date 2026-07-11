from functools import partial
import os
from jax import random
import jax.numpy as np
from jax.scipy.linalg import block_diag
import wandb

from .train_helpers import create_train_state, reduce_lr_on_plateau,\
    linear_warmup, cosine_annealing, constant_lr, train_epoch, validate,\
    save_checkpoint, save_checkpoint_msgpack, load_checkpoint_msgpack,\
    compute_predictor_frobenius
from .dataloading import Datasets
from .seq_model import BatchClassificationModel, RetrievalModel
from .ssm import init_S5SSM
from .ssm_init import make_DPLR_HiPPO
from .mambino_ssm import init_MambinoSSM


def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    if args.USE_WANDB:
        # Make wandb config dictionary
        wandb.init(project=args.wandb_project, job_type='model_training', config=vars(args), entity=args.wandb_entity)
    else:
        wandb.init(mode='offline')

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)
    wandb.log({"block_size": block_size})

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    create_dataset_fn = Datasets[args.dataset]

    # Dataset dependent logic
    if args.dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
        padded = True
        if args.dataset in ["aan-classification"]:
            # Use retreival model for document matching
            retrieval = True
            print("Using retrieval model for document matching")
        else:
            retrieval = False

    else:
        padded = False
        retrieval = False

    # For speech dataset
    if args.dataset in ["speech35-classification"]:
        speech = True
        print("Will evaluate on both resolutions for speech task")
    else:
        speech = False

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
      create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    print(f"[*] Starting S5 Training on `{args.dataset}` =>> Initializing...")

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))

    # ── Mambino-SSM route: replace S5SSM with MambinoSSM (predictor branch
    # + additive PC W_eps) when --use_mambino_ssm is set.  Drop-in
    # compatible signature so all S5 downstream code is unchanged.
    if getattr(args, 'use_mambino_ssm', False):
        bidir_predictor = getattr(args, 'bidir_predictor', False)
        print(f"[*] Using MambinoSSM (proprioceptive predictor + W_eps additive PC)"
              f"{' [BIDIRECTIONAL PREDICTOR]' if bidir_predictor else ''}")
        ssm_init_fn = init_MambinoSSM(H=args.d_model,
                                       P=ssm_size,
                                       Lambda_re_init=Lambda.real,
                                       Lambda_im_init=Lambda.imag,
                                       V=V,
                                       Vinv=Vinv,
                                       C_init=args.C_init,
                                       discretization=args.discretization,
                                       dt_min=args.dt_min,
                                       dt_max=args.dt_max,
                                       conj_sym=args.conj_sym,
                                       clip_eigs=args.clip_eigs,
                                       bidirectional=args.bidirectional,
                                       bidir_predictor=bidir_predictor)
    else:
        ssm_init_fn = init_S5SSM(H=args.d_model,
                                 P=ssm_size,
                                 Lambda_re_init=Lambda.real,
                                 Lambda_im_init=Lambda.imag,
                                 V=V,
                                 Vinv=Vinv,
                                 C_init=args.C_init,
                                 discretization=args.discretization,
                                 dt_min=args.dt_min,
                                 dt_max=args.dt_max,
                                 conj_sym=args.conj_sym,
                                 clip_eigs=args.clip_eigs,
                                 bidirectional=args.bidirectional)

    if retrieval:
        # Use retrieval head for AAN task
        print("Using Retrieval head for {} task".format(args.dataset))
        model_cls = partial(
            RetrievalModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    else:
        model_cls = partial(
            BatchClassificationModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
            glu_rank=getattr(args, 'glu_rank', 0),
            glu_structure=getattr(args, 'glu_structure', 'dense'),
            glu_monarch_heads=getattr(args, 'glu_monarch_heads', 3),
            glu_blockdiag_blocks=getattr(args, 'glu_blockdiag_blocks', 2),
        )

    # initialize training state
    state = create_train_state(model_cls,
                               init_rng,
                               padded,
                               retrieval,
                               in_dim=in_dim,
                               bsz=args.bsz,
                               seq_len=seq_len,
                               weight_decay=args.weight_decay,
                               batchnorm=args.batchnorm,
                               opt_config=args.opt_config,
                               ssm_lr=ssm_lr,
                               lr=lr,
                               dt_global=args.dt_global)

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    steps_per_epoch = int(train_size/args.bsz)
    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        if epoch < args.warmup_end:
            print("using linear warmup for epoch {}".format(epoch+1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end

        elif args.cosine_anneal:
            print("using cosine annealing for epoch {}".format(epoch+1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
        else:
            print("using constant lr for epoch {}".format(epoch+1))
            decay_function = constant_lr
            end_step = None

        # TODO: Switch to letting Optax handle this.
        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)

        train_rng, skey = random.split(train_rng)
        state, train_loss, step, epoch_metrics = train_epoch(
            state,
            skey,
            model_cls,
            trainloader,
            seq_len,
            in_dim,
            args.batchnorm,
            lr_params,
            lambda_pc=getattr(args, 'lambda_pc', 0.0),
        )

        # ── Mambino-style predictor-pathway telemetry ──
        # Format matches ncb/v04_train.py:1889-1897:
        #   E{epoch} ... intr {L_int:.4f}  W_eps {mean:.4f}  C_s {mean:.4f}
        # Adds B_s (predictor-input) frobenius + per-block L_int space-string.
        # No-op for vanilla S5 (empty per_block dict; Frobenius dict lacks
        # predictor keys -> Nones, gracefully skipped).
        per_block_L_int = epoch_metrics.get("per_block_L_int", {})
        intrinsic_eval = epoch_metrics.get("intrinsic_loss", 0.0)
        task_loss_epoch = epoch_metrics.get("task_loss", float(train_loss))
        frob_report = compute_predictor_frobenius(state.params)
        # Aggregate mean-across-blocks for the print line
        def _mean_across(key):
            vals = [b[key] for b in frob_report.values() if key in b]
            return (sum(vals) / len(vals)) if vals else None
        weps_frob_mean = _mean_across("W_eps")
        cs_frob_mean = _mean_across("C_s")
        bs_frob_mean = _mean_across("B_s")
        b_frob_mean = _mean_across("B")
        weps_over_b_mean = _mean_across("W_eps_over_B")
        # Per-block L_int as space-separated string (sorted by layer index)
        if per_block_L_int:
            sorted_items = sorted(
                per_block_L_int.items(),
                key=lambda kv: int(kv[0].split("_")[-1]))
            block_intrinsics_str = " ".join(f"{v:.4g}" for _, v in sorted_items)
        else:
            block_intrinsics_str = ""
        if intrinsic_eval > 0 or weps_frob_mean is not None:
            def _fmt(x):
                return f"{x:.4f}" if x is not None else "n/a"
            print(
                f"[Mambino] E{epoch + 1}  task {task_loss_epoch:.4f}  "
                f"intr {intrinsic_eval:.4f}  "
                f"W_eps {_fmt(weps_frob_mean)}  C_s {_fmt(cs_frob_mean)}  "
                f"B_s {_fmt(bs_frob_mean)}  B {_fmt(b_frob_mean)}  "
                f"W_eps/B {_fmt(weps_over_b_mean)}"
            )
            if block_intrinsics_str:
                print(f"[Mambino] E{epoch + 1}  per_block_L_int: {block_intrinsics_str}")

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = validate(state,
                                         model_cls,
                                         valloader,
                                         seq_len,
                                         in_dim,
                                         args.batchnorm)

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(state,
                                           model_cls,
                                           testloader,
                                           seq_len,
                                           in_dim,
                                           args.batchnorm)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} --Test Loss: {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
            )

        else:
            # else use test set as validation set (e.g. IMDB)
            print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = validate(state,
                                         model_cls,
                                         testloader,
                                         seq_len,
                                         in_dim,
                                         args.batchnorm)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} --"
                f" Test Accuracy: {val_acc:.4f}"
            )

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

            # ── Save BEST checkpoint on every val-acc improvement ──
            # Path controlled by --ckpt_dir.  If --ckpt_dir is empty,
            # skip saving (preserves old S5 behavior).  Saves BOTH the
            # legacy pickle format (for backward compat / debugging) AND
            # the flax msgpack format (which round-trips correctly).
            ckpt_dir = getattr(args, 'ckpt_dir', '') or ''
            if ckpt_dir:
                best_path = os.path.join(ckpt_dir, "best.pkl")
                save_checkpoint(state, best_path, epoch=epoch,
                                test_acc=best_test_acc,
                                test_loss=best_test_loss,
                                args_dict=vars(args),
                                batchnorm=args.batchnorm)
                print(f"[*] Saved best pickle -> {best_path} (test_acc={best_test_acc:.4f})")
                # ── Msgpack format for reliable reload ──
                best_msgpack_base = os.path.join(ckpt_dir, "best")
                save_checkpoint_msgpack(state, best_msgpack_base, epoch=epoch,
                                        test_acc=best_test_acc,
                                        test_loss=best_test_loss,
                                        args_dict=vars(args),
                                        batchnorm=args.batchnorm)
                print(f"[*] Saved best msgpack -> {best_msgpack_base}.msgpack")

            # Do some validation on improvement.
            if speech:
                # Evaluate on resolution 2 val and test sets
                print(f"[*] Running Epoch {epoch + 1} Res 2 Validation...")
                val2_loss, val2_acc = validate(state,
                                               model_cls,
                                               aux_dataloaders['valloader2'],
                                               int(seq_len // 2),
                                               in_dim,
                                               args.batchnorm,
                                               step_rescale=2.0)

                print(f"[*] Running Epoch {epoch + 1} Res 2 Test...")
                test2_loss, test2_acc = validate(state, model_cls, aux_dataloaders['testloader2'], int(seq_len // 2), in_dim, args.batchnorm, step_rescale=2.0)
                print(f"\n=>> Epoch {epoch + 1} Res 2 Metrics ===")
                print(
                    f"\tVal2 Loss: {val2_loss:.5f} --Test2 Loss: {test2_loss:.5f} --"
                    f" Val Accuracy: {val2_acc:.4f}"
                    f" Test Accuracy: {test2_acc:.4f}"
                )

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min)

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        if valloader is not None:
            if speech:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        "Val2 loss": val2_loss,
                        "Val2 Accuracy": val2_acc,
                        "Test2 Loss": test2_loss,
                        "Test2 Accuracy": test2_acc,
                        "count": count,
                        "Learning rate count": lr_count,
                        "Opt acc": opt_acc,
                        "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                        "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                    }
                )
            else:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        "count": count,
                        "Learning rate count": lr_count,
                        "Opt acc": opt_acc,
                        "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                        "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                    }
                )

        else:
            wandb.log(
                {
                    "Training Loss": train_loss,
                    "Val loss": val_loss,
                    "Val Accuracy": val_acc,
                    "count": count,
                    "Learning rate count": lr_count,
                    "Opt acc": opt_acc,
                    "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                    "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                }
            )
        wandb.run.summary["Best Val Loss"] = best_loss
        wandb.run.summary["Best Val Accuracy"] = best_acc
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        wandb.run.summary["Best Test Accuracy"] = best_test_acc

        # ── Mambino predictor-pathway telemetry to wandb ──
        # Log aggregates + per-block breakdown so we can plot L_int(l, epoch)
        # and Frobenius trajectories across epochs.  All keys namespaced
        # under "mambino/" for easy filtering.
        if intrinsic_eval > 0 or weps_frob_mean is not None:
            mambino_log = {
                "mambino/task_loss": task_loss_epoch,
                "mambino/intrinsic_loss": intrinsic_eval,
            }
            if weps_frob_mean is not None:
                mambino_log["mambino/W_eps_frob_mean"] = weps_frob_mean
            if cs_frob_mean is not None:
                mambino_log["mambino/C_s_frob_mean"] = cs_frob_mean
            if bs_frob_mean is not None:
                mambino_log["mambino/B_s_frob_mean"] = bs_frob_mean
            if b_frob_mean is not None:
                mambino_log["mambino/B_frob_mean"] = b_frob_mean
            if weps_over_b_mean is not None:
                mambino_log["mambino/W_eps_over_B_mean"] = weps_over_b_mean
            # Per-block L_int
            for lk, v in per_block_L_int.items():
                mambino_log[f"mambino/L_int/{lk}"] = float(v)
            # Per-block Frobenius (predictor-branch matrices only)
            for lk, sub in frob_report.items():
                for pkey in ("B_s", "C_s", "W_eps",
                             "Lambda_s_abs", "W_eps_over_B"):
                    if pkey in sub:
                        mambino_log[f"mambino/frob/{lk}/{pkey}"] = sub[pkey]
            wandb.log(mambino_log)

        # ── Save FINAL checkpoint every epoch (overwrites previous).  ──
        # This guarantees that on early-stop / SLURM timeout / crash we
        # always have the LAST epoch's state to resume from.  Best ckpt
        # is saved separately above when val_acc improves.  Dual-format
        # (pickle + msgpack) for reliable reload.
        ckpt_dir = getattr(args, 'ckpt_dir', '') or ''
        if ckpt_dir:
            final_path = os.path.join(ckpt_dir, "final.pkl")
            current_test_acc = test_acc if valloader is not None else val_acc
            current_test_loss = test_loss if valloader is not None else val_loss
            save_checkpoint(state, final_path, epoch=epoch,
                            test_acc=current_test_acc,
                            test_loss=current_test_loss,
                            args_dict=vars(args),
                            batchnorm=args.batchnorm)
            final_msgpack_base = os.path.join(ckpt_dir, "final")
            save_checkpoint_msgpack(state, final_msgpack_base, epoch=epoch,
                                    test_acc=current_test_acc,
                                    test_loss=current_test_loss,
                                    args_dict=vars(args),
                                    batchnorm=args.batchnorm)

        if count > args.early_stop_patience:
            break

    # ═══════════════════════════════════════════════════════════════════
    # POST-TRAINING: chip eval sweep + reload verification
    # ═══════════════════════════════════════════════════════════════════
    ckpt_dir = getattr(args, 'ckpt_dir', '') or ''
    chip_sigmas = getattr(args, 'chip_eval_sigmas', '') or ''
    chip_bits = getattr(args, 'chip_eval_bits', '') or ''

    if chip_sigmas or chip_bits:
        sigmas = [float(s) for s in chip_sigmas.split(',')] if chip_sigmas else [0.0]
        bits_list = [int(b) for b in chip_bits.split(',')] if chip_bits else [0]

        print("\n" + "=" * 70)
        print("POST-TRAINING RELOAD VERIFICATION + CHIP EVAL SWEEP")
        print("=" * 70)

        # ── 1) Verify current live state gives expected test_acc ──
        print(f"\n[chip_verify] Re-running final test with LIVE state...")
        _, live_test_acc = validate(state, model_cls, testloader,
                                    seq_len, in_dim, args.batchnorm)
        print(f"[chip_verify] live_test_acc = {live_test_acc:.4f}")

        # ── 2) Msgpack round-trip verify ──
        if ckpt_dir:
            print(f"\n[chip_verify] Reloading msgpack final ckpt...")
            try:
                reloaded_state, meta = load_checkpoint_msgpack(
                    os.path.join(ckpt_dir, "final"), state)
                _, reload_test_acc = validate(reloaded_state, model_cls, testloader,
                                              seq_len, in_dim, args.batchnorm)
                print(f"[chip_verify] reload_test_acc = {reload_test_acc:.4f}")
                print(f"[chip_verify] delta live-vs-reload = "
                      f"{abs(live_test_acc - reload_test_acc):.4f}")
                if abs(live_test_acc - reload_test_acc) < 0.01:
                    print(f"[chip_verify] PASS -- msgpack round-trip works")
                else:
                    print(f"[chip_verify] FAIL -- msgpack round-trip is broken too")
            except Exception as e:
                print(f"[chip_verify] Reload FAILED: {e}")

        # ── 3) Chip noise + ADC sweep on LIVE state ──
        # Build a noise-augmented model_cls for each (sigma, bits) combo
        # by re-invoking init_S5SSM/init_MambinoSSM with the appropriate
        # chip knobs, then calling validate() with the SAME state.
        print(f"\n[chip_eval] Sweeping {len(sigmas)} sigmas x "
              f"{len(bits_list)} bit depths on LIVE state...")

        for sigma in sigmas:
            for bits in bits_list:
                # Build noisy ssm_init_fn matching the training config
                # DAC and ADC precision coupled: both driven by `bits`
                # since they share the analog<->digital boundary at each
                # inter-layer crossing.  A chip designed with 6-bit ADC
                # would use a matching 6-bit DAC on the next layer.
                if getattr(args, 'use_mambino_ssm', False):
                    n_ssm = init_MambinoSSM(H=args.d_model, P=ssm_size,
                        Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
                        V=V, Vinv=Vinv,
                        C_init=args.C_init, discretization=args.discretization,
                        dt_min=args.dt_min, dt_max=args.dt_max,
                        conj_sym=args.conj_sym, clip_eigs=args.clip_eigs,
                        bidirectional=args.bidirectional,
                        bidir_predictor=getattr(args, 'bidir_predictor', False),
                        noise_sigma=sigma, adc_bits=bits, dac_bits=bits)
                else:
                    n_ssm = init_S5SSM(H=args.d_model, P=ssm_size,
                        Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
                        V=V, Vinv=Vinv,
                        C_init=args.C_init, discretization=args.discretization,
                        dt_min=args.dt_min, dt_max=args.dt_max,
                        conj_sym=args.conj_sym, clip_eigs=args.clip_eigs,
                        bidirectional=args.bidirectional,
                        noise_sigma=sigma, adc_bits=bits, dac_bits=bits)
                # Build noisy model_cls with same non-SSM args as training
                if retrieval:
                    n_model_cls = partial(RetrievalModel,
                        ssm=n_ssm, d_output=n_classes, d_model=args.d_model,
                        n_layers=args.n_layers, padded=padded,
                        activation=args.activation_fn, dropout=args.p_dropout,
                        prenorm=args.prenorm, batchnorm=args.batchnorm,
                        bn_momentum=args.bn_momentum)
                else:
                    n_model_cls = partial(BatchClassificationModel,
                        ssm=n_ssm, d_output=n_classes, d_model=args.d_model,
                        n_layers=args.n_layers, padded=padded,
                        activation=args.activation_fn, dropout=args.p_dropout,
                        mode=args.mode, prenorm=args.prenorm,
                        batchnorm=args.batchnorm, bn_momentum=args.bn_momentum,
                        glu_rank=getattr(args, 'glu_rank', 0),
                        glu_structure=getattr(args, 'glu_structure', 'dense'),
                        glu_monarch_heads=getattr(args, 'glu_monarch_heads', 3),
                        glu_blockdiag_blocks=getattr(args, 'glu_blockdiag_blocks', 2))
                # Run validate with noise rng
                # SSM needs an rng whenever it takes the non-fast path,
                # which is triggered by sigma > 0 OR bits > 0.
                nrs = 42 if (sigma > 0 or bits > 0) else None
                v_loss, v_acc = validate(state, n_model_cls, valloader,
                                         seq_len, in_dim, args.batchnorm,
                                         noise_rng_seed=nrs)
                t_loss, t_acc = validate(state, n_model_cls, testloader,
                                         seq_len, in_dim, args.batchnorm,
                                         noise_rng_seed=nrs)
                print(f"[chip_eval] sigma={sigma:.4f}  bits={bits}  "
                      f"val_acc={v_acc:.4f}  test_acc={t_acc:.4f}")
        print(f"\n[chip_eval] Sweep complete.")
