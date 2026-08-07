"""Stage C, module 4 — runner. Builds the English->L2->English stream, extracts each backbone's
features once, and scores frozen vs always-adapt readouts -> per-segment BPC + recovery curve.

The headline: on the L2 segment, does always-adapt PAP recover (low BPC) where the frozen models
(S5 and PAP) stay high? And does it re-adapt when the stream returns to English (EN2)?
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from paper_v2_ppac.pap_streaming import stream as S
from paper_v2_ppac.pap_streaming import backbone as B
from paper_v2_ppac.pap_streaming import adapt as A


def per_segment_bpc(nll, spans):
    """spans align to positions 0..L-1; nll covers 0..L-2 (last has no target). Returns {label: bpc}."""
    out = {}
    for label, a, b in spans:
        b = min(b, len(nll))
        if b > a:
            out[label] = A.bpc(nll[a:b])
    return out


def recovery_curve(nll, bin_bytes=250):
    """Binned BPC vs position (for plotting the recovery)."""
    n = (len(nll) // bin_bytes) * bin_bytes
    binned = nll[:n].reshape(-1, bin_bytes).mean(1) / np.log(2.0)
    centers = (np.arange(len(binned)) + 0.5) * bin_bytes
    return centers, binned


def run_arm(name, model, params, ids, adapt, eta, chunk, spans):
    feat, base = B.extract(model, params, ids[:-1])          # predict ids[1:] from ids[:-1]
    targets = ids[1:]
    nll, dW = A.stream_nll(feat, base, targets, adapt=adapt, eta=eta, chunk=chunk)
    seg = per_segment_bpc(nll, spans)
    tag = f"{name}{'+adapt' if adapt else ' frozen'}"
    print(f"[{tag:16s}] " + "  ".join(f"{k}={v:.3f}" for k, v in seg.items())
          + f"  | overall={A.bpc(nll):.3f}" + (f"  meandW={dW[dW>0].mean():.3f}" if adapt else ""))
    return dict(name=tag, nll=nll, seg=seg, dW=dW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pap_msgpack", required=True); ap.add_argument("--pap_meta", required=True)
    ap.add_argument("--s5_msgpack", default=""); ap.add_argument("--s5_meta", default="")
    ap.add_argument("--eng", required=True); ap.add_argument("--l2", required=True)
    ap.add_argument("--seg_bytes", type=int, default=20000); ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--eta", type=float, default=0.5); ap.add_argument("--eng_offset", type=int, default=0)
    ap.add_argument("--l2_label", default="FR"); ap.add_argument("--out", default="")
    args = ap.parse_args()

    st = S.eng_l2_eng(args.eng, args.l2, args.seg_bytes, eng_offset=args.eng_offset, l2_label=args.l2_label)
    print(f"stream: {len(st['ids'])} bytes  spans={[(l,a,b) for l,a,b in st['spans']]}  chunk={args.chunk} eta={args.eta}")

    results = []
    pap_model, pap_params, _ = B.load_backbone(args.pap_msgpack, args.pap_meta)
    results.append(run_arm("PAP", pap_model, pap_params, st["ids"], False, args.eta, args.chunk, st["spans"]))
    results.append(run_arm("PAP", pap_model, pap_params, st["ids"], True, args.eta, args.chunk, st["spans"]))
    if args.s5_msgpack:
        s5_model, s5_params, _ = B.load_backbone(args.s5_msgpack, args.s5_meta)
        results.append(run_arm("S5", s5_model, s5_params, st["ids"], False, args.eta, args.chunk, st["spans"]))
        results.append(run_arm("S5", s5_model, s5_params, st["ids"], True, args.eta, args.chunk, st["spans"]))

    if args.out:
        np.savez(args.out, spans=np.array(st["spans"], dtype=object),
                 **{r["name"].replace(" ", "_").replace("+", "_"): r["nll"] for r in results})
        print(f"[*] saved {args.out}")


if __name__ == "__main__":
    main()
