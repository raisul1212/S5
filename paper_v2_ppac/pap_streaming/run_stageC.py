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
    """spans are ids-coordinates; nll[t] scores target ids[t+1] -> shift by -1 so each segment's
    BPC covers ITS bytes' predictions (Fable off-by-one fix). nll covers positions 0..L-2."""
    out = {}
    for label, a, b in spans:
        lo, hi = max(a - 1, 0), min(b - 1, len(nll))
        if hi > lo:
            out[label] = A.bpc(nll[lo:hi])
    return out


def recovery_curve(nll, bin_bytes=250):
    """Binned BPC vs position (for plotting the recovery)."""
    n = (len(nll) // bin_bytes) * bin_bytes
    binned = nll[:n].reshape(-1, bin_bytes).mean(1) / np.log(2.0)
    centers = (np.arange(len(binned)) + 0.5) * bin_bytes
    return centers, binned


def run_arm(name, feat, base, targets, adapt, eta, chunk, spans, mode="readout"):
    nll, dW = A.stream_nll(feat, base, targets, adapt=adapt, eta=eta, chunk=chunk, mode=mode)
    seg = per_segment_bpc(nll, spans)
    suff = "+adapt" if adapt else " frozen"
    if adapt and mode == "bias":
        suff = "+bias(ctrl)"
    tag = f"{name}{suff}"
    print(f"[{tag:16s}] " + "  ".join(f"{k}={v:.3f}" for k, v in seg.items())
          + f"  | overall={A.bpc(nll):.3f}" + (f"  meandW={dW[dW>0].mean():.3f}" if adapt else ""))
    return dict(name=tag, nll=nll, seg=seg, dW=dW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pap_msgpack", required=True); ap.add_argument("--pap_meta", required=True)
    ap.add_argument("--s5_msgpack", default=""); ap.add_argument("--s5_meta", default="")
    ap.add_argument("--eng", required=True); ap.add_argument("--l2", required=True)
    ap.add_argument("--seg_bytes", type=int, default=20000); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--eta", type=float, default=0.1)
    # EN drawn from the enwik8 TEST region (>=95M) so it is NOT training data (Fable fix #1);
    # assumes --eng is the full enwik8. Requires 2*seg_bytes <= 5M (ok at 20k).
    ap.add_argument("--eng_offset", type=int, default=95_000_000)
    ap.add_argument("--l2_label", default="FR"); ap.add_argument("--out", default="")
    args = ap.parse_args()

    st = S.eng_l2_eng(args.eng, args.l2, args.seg_bytes, eng_offset=args.eng_offset, l2_label=args.l2_label)
    print(f"stream: {len(st['ids'])} bytes  spans={[(l,a,b) for l,a,b in st['spans']]}  "
          f"chunk={args.chunk} eta={args.eta} eng_offset={args.eng_offset}")

    results = []
    pap_model, pap_params, _ = B.load_backbone(args.pap_msgpack, args.pap_meta)
    pf, pb = B.extract(pap_model, pap_params, st["ids"][:-1]); tgt = st["ids"][1:]
    results.append(run_arm("PAP", pf, pb, tgt, False, args.eta, args.chunk, st["spans"]))
    results.append(run_arm("PAP", pf, pb, tgt, True, args.eta, args.chunk, st["spans"]))
    results.append(run_arm("PAP", pf, pb, tgt, True, args.eta, args.chunk, st["spans"], mode="bias"))  # attribution control
    if args.s5_msgpack:
        s5_model, s5_params, _ = B.load_backbone(args.s5_msgpack, args.s5_meta)
        sf, sb = B.extract(s5_model, s5_params, st["ids"][:-1])
        results.append(run_arm("S5", sf, sb, tgt, False, args.eta, args.chunk, st["spans"]))
        results.append(run_arm("S5", sf, sb, tgt, True, args.eta, args.chunk, st["spans"]))

    if args.out:
        np.savez(args.out, spans=np.array(st["spans"], dtype=object),
                 **{r["name"].replace(" ", "_").replace("+", "_"): r["nll"] for r in results})
        print(f"[*] saved {args.out}")


if __name__ == "__main__":
    main()
