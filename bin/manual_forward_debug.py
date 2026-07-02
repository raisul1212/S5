"""Debug version of manual_forward: print intermediate activations
at each stage to find WHERE the explosion happens.

Runs on 1 sample only.
"""
import argparse
import pickle
import sys
import os

import jax
import jax.numpy as np
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s5.dataloading import Datasets
from s5.train_helpers import prep_batch


def stat(name, x):
    xa = np.asarray(x)
    if xa.dtype in [np.complex64, np.complex128]:
        r = xa.real
        i = xa.imag
        print(f"  {name}: shape={tuple(xa.shape)} complex, "
              f"real:mean={float(r.mean()):.3f} std={float(r.std()):.3f} "
              f"[{float(r.min()):.2f},{float(r.max()):.2f}]  "
              f"imag:mean={float(i.mean()):.3f} std={float(i.std()):.3f}")
    else:
        print(f"  {name}: shape={tuple(xa.shape)} dtype={xa.dtype} "
              f"mean={float(xa.mean()):.4f} std={float(xa.std()):.4f} "
              f"[{float(xa.min()):.4f},{float(xa.max()):.4f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True)
    args = p.parse_args()

    with open(args.ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    print(f"saved test_acc: {ckpt['test_acc']:.4f}")
    params = jax.tree_util.tree_map(np.asarray, ckpt["params"])
    batch_stats = jax.tree_util.tree_map(np.asarray, ckpt["batch_stats"])

    # Dataset
    create_dataset_fn = Datasets["listops-classification"]
    _, _, testloader, _, _, seq_len, in_dim, _ = \
        create_dataset_fn("./raw_datasets", seed=6554595, bsz=50)

    # Grab 1 sample
    for batch in testloader:
        inputs_raw, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        break
    inputs_arr, lengths = inputs_raw

    # Just do 1 sample
    x_in = inputs_arr[0]     # (L, in_dim)
    length = lengths[0]
    label = labels[0]
    print(f"\ninput x shape: {x_in.shape}, length={length}, label={label}")
    stat("x_in", x_in)

    # Encoder Dense
    W_enc = params["encoder"]["encoder"]["kernel"]
    b_enc = params["encoder"]["encoder"]["bias"]
    x = x_in @ W_enc + b_enc  # (L, 128)
    stat("after_encoder", x)

    # Walk through layers manually
    for i in range(8):
        print(f"\n--- Layer {i} ---")
        skip = x
        layer_p = params["encoder"][f"layers_{i}"]
        layer_bs = batch_stats["encoder"][f"layers_{i}"]
        seq_p = layer_p["seq"]

        # SSM
        Lambda = seq_p["Lambda_re"] + 1j * seq_p["Lambda_im"]
        log_step = seq_p["log_step"][:, 0]
        step = np.exp(log_step)
        B_stored = seq_p["B"]
        B_complex = B_stored[..., 0] + 1j * B_stored[..., 1]
        Lambda_bar = np.exp(Lambda * step)
        B_bar = ((1 / Lambda) * (Lambda_bar - 1.0))[..., None] * B_complex

        stat(f"Lambda_bar (layer {i})", Lambda_bar)
        stat(f"B_bar (layer {i})", B_bar)

        C1 = seq_p["C1"][..., 0] + 1j * seq_p["C1"][..., 1]
        C2 = seq_p["C2"][..., 0] + 1j * seq_p["C2"][..., 1]
        D = seq_p["D"]

        L = x.shape[0]
        Bu = jax.vmap(lambda u: B_bar @ u.astype(np.complex64))(x)
        Lambda_elements = Lambda_bar * np.ones((L, Lambda_bar.shape[0]))
        stat(f"Bu (layer {i})", Bu)

        def binary_op(q_i, q_j):
            A_i, b_i = q_i
            A_j, b_j = q_j
            return A_j * A_i, A_j * b_i + b_j

        _, xs_fwd = lax.associative_scan(jax.vmap(binary_op),
                                         (Lambda_elements, Bu))
        _, xs_bwd = lax.associative_scan(jax.vmap(binary_op),
                                         (Lambda_elements, Bu), reverse=True)
        stat(f"xs_fwd (layer {i})", xs_fwd)
        stat(f"xs_bwd (layer {i})", xs_bwd)

        def readout(h_fwd, h_bwd):
            return 2.0 * ((C1 @ h_fwd) + (C2 @ h_bwd)).real
        ys = jax.vmap(readout)(xs_fwd, xs_bwd)
        stat(f"ys (SSM readout, layer {i})", ys)

        Du = D[None, :] * x
        ssm_out = ys + Du
        stat(f"ssm_out = ys + Du (layer {i})", ssm_out)

        # half_glu2
        x1 = jax.nn.gelu(ssm_out)
        stat(f"gelu output (layer {i})", x1)
        W_out2 = layer_p["out2"]["kernel"]
        b_out2 = layer_p["out2"]["bias"]
        gate_pre = x1 @ W_out2 + b_out2
        gate = jax.nn.sigmoid(gate_pre)
        stat(f"gate values (layer {i})", gate)
        gated = ssm_out * gate
        stat(f"gated output (layer {i})", gated)

        x_after_skip = skip + gated
        stat(f"after_skip (layer {i})", x_after_skip)

        # BN
        norm_p = layer_p["norm"]
        scale = norm_p["scale"]
        bias = norm_p["bias"]
        running_mean = layer_bs["norm"]["mean"]
        running_var = layer_bs["norm"]["var"]
        stat(f"running_mean (layer {i})", running_mean)
        stat(f"running_var (layer {i})", running_var)
        stat(f"scale (layer {i})", scale)

        x_normed = (x_after_skip - running_mean) / np.sqrt(running_var + 1e-5)
        stat(f"after_bn_normalize (layer {i})", x_normed)
        x = x_normed * scale + bias
        stat(f"after_bn_full (layer {i}) [final layer output]", x)

    # Pool + decode
    print("\n--- Pool + Decode ---")
    mask = (np.arange(x.shape[0]) < length).astype(x.dtype)[:, None]
    pooled = np.sum(x * mask, axis=0) / length
    stat("pooled", pooled)
    W_dec = params["decoder"]["kernel"]
    b_dec = params["decoder"]["bias"]
    logits = pooled @ W_dec + b_dec
    stat("logits (pre-softmax)", logits)
    pred = int(np.argmax(logits))
    print(f"\npredicted class: {pred}, true label: {label}")


if __name__ == "__main__":
    main()
