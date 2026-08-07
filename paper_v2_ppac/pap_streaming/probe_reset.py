"""Stage C state-reset probe. Washout vs accumulation for PAP's within-L2 improvement.

Run each model on the L2 segment in independent windows of size k, each from ZERO state.
Per-window loss binned by position-WITHIN-window: if it drops from high (pos 0, no context)
to low (pos k-1, k tokens of context) and the late-window loss ~= the no-reset late-segment
loss, then the improvement is state relaxation over ~forgetting-horizon tokens (washout / local
in-context context-fill), NOT long-range accumulation. Try several k to read the timescale.
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from paper_v2_ppac.pap_streaming import stream as S, backbone as B

LN2 = np.log(2.0)


def bpc_of(base, tgt):
    m = base.max(-1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(base - m).sum(-1))
    return (lse - base[np.arange(len(tgt)), tgt]) / LN2          # per-position BPC


def noreset_curve(model, params, ids, la, lb):
    _, base = B.extract(model, params, ids[:-1])
    l = bpc_of(np.asarray(base), ids[1:])[max(la - 1, 0):lb - 1]
    t = len(l) // 3
    return float(l[:t].mean()), float(l[-t:].mean())            # early-third, late-third


def reset_curve(model, params, seg_ids, k):
    """Mean BPC vs position-within-window over independent zero-state k-windows."""
    per = [[] for _ in range(k - 1)]
    for w0 in range(0, len(seg_ids) - 1, k):
        win = seg_ids[w0:w0 + k]
        if len(win) < 8:
            continue
        _, base = B.extract(model, params, win[:-1])
        l = bpc_of(np.asarray(base), win[1:])
        for i in range(len(l)):
            per[i].append(l[i])
    curve = np.array([np.mean(x) for x in per if x])
    f = max(1, len(curve) // 10)
    return curve, float(curve[:f].mean()), float(curve[-f:].mean())   # curve, first-10%, last-10%


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pap_msgpack", required=True); ap.add_argument("--pap_meta", required=True)
    ap.add_argument("--s5_msgpack", required=True); ap.add_argument("--s5_meta", required=True)
    ap.add_argument("--eng", required=True); ap.add_argument("--l2", required=True)
    ap.add_argument("--seg", type=int, default=8000); ap.add_argument("--eng_base", type=int, default=95_000_000)
    ap.add_argument("--ks", default="250,1000,4000"); ap.add_argument("--l2_label", default="RU")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    en = S.load_bytes(args.eng, 2 * args.seg, args.eng_base)
    l2 = S.load_bytes(args.l2, args.seg, 0)
    st = S.build_stream([("EN", en[:args.seg]), (args.l2_label, l2), ("EN2", en[args.seg:])])
    ids = st["ids"]; la, lb = st["spans"][1][1], st["spans"][1][2]
    seg = ids[la:lb]

    print(f"\n==== state-reset probe ({args.l2_label}, seg={args.seg}) ====")
    for name, ck in [("PAP-error", (args.pap_msgpack, args.pap_meta)),
                     ("S5", (args.s5_msgpack, args.s5_meta))]:
        model, params = B.load_backbone(*ck)[:2]
        ne, nl = noreset_curve(model, params, ids, la, lb)
        print(f"\n{name}:")
        print(f"  no-reset (English-primed): {args.l2_label} early-third {ne:.3f} -> late-third {nl:.3f} BPC")
        for k in ks:
            _, we, wl = reset_curve(model, params, seg, k)
            note = ""
            if name == "PAP-error":
                note = ("  <- reset late ~= no-reset late => improvement is CONTEXT-FILL over <=k tokens (washout)"
                        if abs(wl - nl) < 0.08 else "  <- reset late != no-reset late => needs >k context")
            print(f"  reset k={k:5d}: within-window first-10% {we:.3f} -> last-10% {wl:.3f} BPC{note}")
    print("\nRESET_DONE")


if __name__ == "__main__":
    main()
