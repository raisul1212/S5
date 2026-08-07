"""Stage C mechanism probes (no retraining). For PAP-error / PAP-input / S5 over an EN->L2->EN
stream, resolve the within-L2 loss change into its cause, using only the frozen base logits:

  - loss(t), entropy(t), top1-acc(t) per token; early-third vs late-third of the L2 segment.
  - decision:
      S5   loss UP + entropy DOWN  => confidently-wrong (HiPPO state drifts, not "fails to adapt")
      PAP  loss DOWN + acc UP      => genuine gain (adaptation or retrieval)
      PAP  loss DOWN + entropy UP + acc flat => FLATTENING ARTIFACT (RMSNorm caps peakiness)
  - bigram floor: BPC of an order-1 byte model fit on the L2 segment -> is PAP's plateau just
    low-order statistics?
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from paper_v2_ppac.pap_streaming import stream as S, backbone as B

LN2 = np.log(2.0)


def _softmax_stats(base, tgt):
    m = base.max(-1, keepdims=True)
    z = np.exp(base - m); p = z / z.sum(-1, keepdims=True)
    lp = np.log(p + 1e-12)
    loss = -lp[np.arange(len(tgt)), tgt] / LN2                     # BPC per token
    ent = -(p * lp).sum(-1) / LN2                                  # entropy (bits) per token
    acc = (base.argmax(-1) == tgt).astype(np.float32)
    return loss, ent, acc


def bigram_bpc(fit_ids, eval_ids):
    """Order-1 byte model: fit P(b_{t+1}|b_t) on a large fit corpus, eval BPC on the segment."""
    V = 256
    C = np.ones((V, V))                                            # Laplace smoothing
    np.add.at(C, (fit_ids[:-1], fit_ids[1:]), 1.0)
    P = C / C.sum(1, keepdims=True)
    return float(np.mean(-np.log2(P[eval_ids[:-1], eval_ids[1:]])))


def thirds(x):
    t = max(1, len(x) // 3)
    return float(np.mean(x[:t])), float(np.mean(x[-t:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pap_msgpack", required=True); ap.add_argument("--pap_meta", required=True)
    ap.add_argument("--pap_inp_msgpack", required=True); ap.add_argument("--pap_inp_meta", required=True)
    ap.add_argument("--s5_msgpack", required=True); ap.add_argument("--s5_meta", required=True)
    ap.add_argument("--eng", required=True); ap.add_argument("--l2", required=True)
    ap.add_argument("--seg", type=int, default=8000); ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--eng_base", type=int, default=95_000_000); ap.add_argument("--l2_label", default="RU")
    args = ap.parse_args()

    bks = [("PAP-error", B.load_backbone(args.pap_msgpack, args.pap_meta)[:2]),
           ("PAP-input", B.load_backbone(args.pap_inp_msgpack, args.pap_inp_meta)[:2]),
           ("S5", B.load_backbone(args.s5_msgpack, args.s5_meta)[:2])]

    agg = {name: {k: [] for k in ("l_e", "l_l", "e_e", "e_l", "a_e", "a_l")} for name, _ in bks}
    l2_fit = S.load_bytes(args.l2, 500_000, 1_000_000)             # large DISJOINT L2 chunk for the bigram floor
    bfloor_list = []
    for s in range(args.n):
        en = S.load_bytes(args.eng, 2 * args.seg, args.eng_base + s * (3 * args.seg + 5000))
        l2 = S.load_bytes(args.l2, args.seg, s * (2 * args.seg + 3000))
        st = S.build_stream([("EN", en[:args.seg]), (args.l2_label, l2), ("EN2", en[args.seg:])])
        ids = st["ids"]; tgt = ids[1:]; la, lb = st["spans"][1][1], st["spans"][1][2]
        sl = slice(max(la - 1, 0), lb - 1)                         # L2 rows of the (t->t+1) arrays
        bfloor_list.append(bigram_bpc(l2_fit, ids[la:lb]))
        for name, (model, params) in bks:
            _, base = B.extract(model, params, ids[:-1])
            loss, ent, acc = _softmax_stats(np.asarray(base), tgt)
            le, ll = thirds(loss[sl]); ee, el = thirds(ent[sl]); ae, al = thirds(acc[sl])
            d = agg[name]
            d["l_e"].append(le); d["l_l"].append(ll); d["e_e"].append(ee)
            d["e_l"].append(el); d["a_e"].append(ae); d["a_l"].append(al)

    bf = float(np.mean(bfloor_list))
    print(f"\n==== Stage-C mechanism probe ({args.l2_label}, n={args.n}, seg={args.seg}) ====")
    print(f"order-1 bigram floor (in-sample) on the {args.l2_label} segment: {bf:.3f} BPC\n")
    print(f"{'model':10s} | {'BPC early->late':>18s} | {'entropy(bits) e->l':>19s} | {'top1-acc e->l':>16s} | verdict")
    for name, _ in bks:
        d = agg[name]; f = lambda k: float(np.mean(d[k]))
        le, ll, ee, el, ae, al = f("l_e"), f("l_l"), f("e_e"), f("e_l"), f("a_e"), f("a_l")
        dl, de, da = ll - le, el - ee, al - ae
        if dl > 0.02:   v = "loss UP" + ("  + entropy DOWN => CONFIDENTLY-WRONG" if de < -0.02 else "")
        elif dl < -0.02:
            if da > 0.01: v = "loss DOWN + ACC UP => genuine gain"
            elif de > 0.02: v = "loss DOWN + ENTROPY UP, acc flat => FLATTENING ARTIFACT"
            else: v = "loss DOWN, acc~flat, entropy~flat => washout/other"
        else: v = "~flat"
        print(f"{name:10s} | {le:6.3f} -> {ll:6.3f} ({dl:+.3f}) | {ee:6.3f} -> {el:6.3f} ({de:+.2f}) "
              f"| {ae:5.3f} -> {al:5.3f} ({da:+.3f}) | {v}")
    print(f"\n(PAP plateau vs bigram floor {bf:.3f}: if close, the within-segment 'gain' is order-1 stats.)")
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
