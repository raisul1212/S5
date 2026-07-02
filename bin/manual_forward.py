"""Manual forward pass using ONLY raw numpy arrays from the checkpoint.
NO Flax modules, NO nn.vmap, NO state.replace, NO module class magic.

Just: load params dict, walk through the S5 architecture step by step,
compute layer outputs, verify accuracy matches saved test_acc.

If this WORKS while the Flax-based path FAILS, we've isolated the bug
to Flax's module system.  Also gives us a workable eval path for the
chip noise/quant experiments -- we can inject noise directly into the
manual forward pass.
"""
import argparse
import pickle
import sys
import os

import jax
import jax.numpy as np
from jax import random, lax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s5.dataloading import Datasets
from s5.train_helpers import prep_batch


def ssm_layer_forward(x_seq, lengths, layer_p, layer_bs, step_rescale=1.0):
    """Apply one S5 layer manually.
    x_seq: (L, H) float
    layer_p: params dict for encoder/layers_i
    layer_bs: batch_stats dict for encoder/layers_i (only 'norm' key)
    Returns: (L, H) float after skip + BN.
    """
    skip = x_seq
    L, H = x_seq.shape

    # ── SSM (seq) submodule ──────────────────────────────────────
    seq_p = layer_p["seq"]
    Lambda_re = seq_p["Lambda_re"]  # (P,)
    Lambda_im = seq_p["Lambda_im"]  # (P,)
    Lambda = Lambda_re + 1j * Lambda_im  # (P,) complex

    log_step = seq_p["log_step"][:, 0]  # (P,)
    step = step_rescale * np.exp(log_step)  # (P,)

    # B: stored as (P, H, 2) real+imag
    B_stored = seq_p["B"]                       # (P, H, 2)
    B_complex = B_stored[..., 0] + 1j * B_stored[..., 1]  # (P, H)

    # ZOH discretization
    Lambda_bar = np.exp(Lambda * step)           # (P,)
    B_bar = ((1 / Lambda) * (Lambda_bar - 1.0))[..., None] * B_complex  # (P, H)

    # C_tilde: forward + backward (bidirectional)
    C1_stored = seq_p["C1"]                     # (H, P, 2)
    C2_stored = seq_p["C2"]                     # (H, P, 2)
    C1 = C1_stored[..., 0] + 1j * C1_stored[..., 1]  # (H, P) complex
    C2 = C2_stored[..., 0] + 1j * C2_stored[..., 1]  # (H, P) complex
    # bidirectional: concatenate C1 and C2 along P axis for reading
    # concatenated 2P state = [fwd_state ; bwd_state]

    D = seq_p["D"]                              # (H,)

    # Parallel scan: forward
    Bu = jax.vmap(lambda u: B_bar @ u.astype(np.complex64))(x_seq)  # (L, P) complex
    Lambda_elements = Lambda_bar * np.ones((L, Lambda_bar.shape[0]))  # (L, P)

    def binary_op(q_i, q_j):
        A_i, b_i = q_i
        A_j, b_j = q_j
        return A_j * A_i, A_j * b_i + b_j

    _, xs_fwd = lax.associative_scan(jax.vmap(binary_op),
                                     (Lambda_elements, Bu))
    _, xs_bwd = lax.associative_scan(jax.vmap(binary_op),
                                     (Lambda_elements, Bu), reverse=True)
    # xs_fwd, xs_bwd: (L, P) complex

    # Apply C: 2 * Re(C1 @ h_fwd + C2 @ h_bwd)  (conj_sym=True implies 2x)
    def readout(h_fwd, h_bwd):
        return 2.0 * ((C1 @ h_fwd) + (C2 @ h_bwd)).real
    ys = jax.vmap(readout)(xs_fwd, xs_bwd)      # (L, H)

    # + D * x
    Du = D[None, :] * x_seq
    ssm_out = ys + Du                            # (L, H)

    # ── half_glu2 gate ──────────────────────────────────────────
    # x1 = drop(gelu(ssm_out))
    x1 = jax.nn.gelu(ssm_out)
    # gate = sigmoid(out2 @ x1)
    W_out2 = layer_p["out2"]["kernel"]           # (H, H)
    b_out2 = layer_p["out2"]["bias"]             # (H,)
    gate_pre = x1 @ W_out2 + b_out2              # (L, H)
    gated = ssm_out * jax.nn.sigmoid(gate_pre)   # (L, H)

    # ── Skip connection ─────────────────────────────────────────
    x_after_skip = skip + gated                  # (L, H)

    # ── BatchNorm with running stats ────────────────────────────
    norm_p = layer_p["norm"]
    scale = norm_p["scale"]                      # (H,)
    bias = norm_p["bias"]                        # (H,)
    running_mean = layer_bs["norm"]["mean"]      # (H,)
    running_var = layer_bs["norm"]["var"]        # (H,)
    # BN over feature dim per token
    x_normed = (x_after_skip - running_mean) / np.sqrt(running_var + 1e-5)
    x_out = x_normed * scale + bias
    return x_out


def full_forward(params, batch_stats, inputs_onehot, lengths):
    """Manual full forward pass for one sequence.

    inputs_onehot: (L, in_dim) one-hot
    lengths: scalar length
    Returns: (n_classes,) log-softmax logits
    """
    # Encoder Dense
    W_enc = params["encoder"]["encoder"]["kernel"]  # (in_dim, d_model)
    b_enc = params["encoder"]["encoder"]["bias"]    # (d_model,)
    x = inputs_onehot @ W_enc + b_enc               # (L, d_model)

    # 8 SSM layers
    for i in range(8):
        layer_p = params["encoder"][f"layers_{i}"]
        layer_bs = batch_stats["encoder"][f"layers_{i}"]
        x = ssm_layer_forward(x, lengths, layer_p, layer_bs)

    # Masked mean pool
    mask = (np.arange(x.shape[0]) < lengths).astype(x.dtype)[:, None]
    pooled = np.sum(x * mask, axis=0) / lengths  # (d_model,)

    # Decoder
    W_dec = params["decoder"]["kernel"]  # (d_model, n_classes)
    b_dec = params["decoder"]["bias"]    # (n_classes,)
    logits = pooled @ W_dec + b_dec  # (n_classes,)
    return jax.nn.log_softmax(logits, axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--max_batches", type=int, default=None,
                   help="If set, only eval first N batches")
    args = p.parse_args()

    print(f"Loading {args.ckpt_path}")
    with open(args.ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    print(f"saved test_acc: {ckpt['test_acc']:.4f}")

    params = jax.tree_util.tree_map(np.asarray, ckpt["params"])
    batch_stats = jax.tree_util.tree_map(np.asarray, ckpt["batch_stats"])

    # Dataset
    create_dataset_fn = Datasets["listops-classification"]
    _, valloader, testloader, _, _, seq_len, in_dim, _ = \
        create_dataset_fn("./raw_datasets", seed=6554595, bsz=50)

    # Vmap the full forward over batch
    batched_forward = jax.jit(jax.vmap(
        full_forward, in_axes=(None, None, 0, 0)))

    total_correct, total_count = 0, 0
    for bi, batch in enumerate(testloader):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        inputs_raw, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        # inputs_raw is (inputs_arr, lengths) for padded=True
        inputs_arr, lengths = inputs_raw
        # inputs_arr: (batch, seq_len, in_dim)  already one-hot from prep_batch

        logits = batched_forward(params, batch_stats, inputs_arr, lengths)
        preds = np.argmax(logits, axis=-1)
        correct = int(np.sum(preds == labels))
        total_correct += correct
        total_count += len(labels)
        if bi < 2:
            print(f"  batch {bi}: {correct}/{len(labels)} = {correct/len(labels):.4f}")
            print(f"    logits shape={logits.shape} mean={float(logits.mean()):.2f} std={float(logits.std()):.2f}")
    acc = total_correct / total_count
    print(f"\nManual forward pass test acc: {acc:.4f}")
    print(f"Saved test_acc:                 {ckpt['test_acc']:.4f}")
    print(f"Delta:                          {abs(acc - ckpt['test_acc']):.4f}")
    if abs(acc - ckpt['test_acc']) < 0.01:
        print("PASS -- manual forward reproduces training's saved test_acc")
    else:
        print("FAIL -- manual forward doesn't match")


if __name__ == "__main__":
    main()
