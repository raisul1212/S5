"""Stage C RIGOROUS verification driver. For each of N disjoint stream samples and each
condition (FR / RU / EN-noshift), extracts PAP & S5 features once and scores every arm:
  frozen | adapt(eta sweep) | bias-only(ctrl) | random-features(ctrl)
Aggregates L2-segment BPC as mean +/- std across samples, saves per-position recovery curves
(incl. FROZEN curves -> the in-context-adaptation probe), and prints a summary table.

Everything here is CPU-cheap (frozen backbone extracted once/sample). MULTI-SEED (different
training seeds of PAP & S5) is the one axis this does NOT cover -> needs GPU, run separately.
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from paper_v2_ppac.pap_streaming import stream as S, backbone as B, adapt as A
from paper_v2_ppac.pap_streaming.run_stageC import per_segment_bpc, recovery_curve


def _seg_stream(eng_path, mid_path, seg, eng_off, mid_off, mid_label, mid_is_eng):
    """EN[eng_off] -> MID[mid_off] -> EN2[eng_off+seg]. If mid_is_eng, MID is a disjoint English slice."""
    en = S.load_bytes(eng_path, 2 * seg, eng_off)
    if mid_is_eng:
        mid = S.load_bytes(eng_path, seg, eng_off + 2 * seg)   # 3rd disjoint English slice
    else:
        mid = S.load_bytes(mid_path, seg, mid_off)
    return S.build_stream([("EN", en[:seg]), (mid_label, mid), ("EN2", en[seg:2 * seg])])


def run_condition(backbones, eng, midpath, seg, n, chunk, etas, mid_label, mid_is_eng, en_base):
    """Returns per-arm arrays of L2-segment BPC over n samples + averaged recovery curves."""
    rng = np.random.RandomState(1234)
    arms = ["frozen", "bias", "random"] + [f"adapt_e{e}" for e in etas]
    L2 = mid_label
    acc = {bk: {arm: [] for arm in arms} for bk, _ in backbones}
    probe = {bk: {"early": [], "late": []} for bk, _ in backbones}   # FROZEN L2 early/late thirds
    curves = {bk: {} for bk, _ in backbones}      # arm -> list of (centers, binned) for L2 region
    for s in range(n):
        eoff = en_base + s * (3 * seg + 5000)
        moff = s * (2 * seg + 3000)
        st = _seg_stream(eng, midpath, seg, eoff, moff, L2, mid_is_eng)
        ids = st["ids"]; tgt = ids[1:]; spans = st["spans"]
        la, lb = spans[1][1], spans[1][2]                     # L2 segment [la, lb)
        for bk, (model, params) in backbones:
            feat, base = B.extract(model, params, ids[:-1])
            randf = rng.standard_normal(feat.shape).astype(np.float32)
            runs = {
                "frozen": A.stream_nll(feat, base, tgt, adapt=False, chunk=chunk),
                "bias":   A.stream_nll(feat, base, tgt, adapt=True, chunk=chunk, mode="bias", eta=0.1),
                "random": A.stream_nll(randf, base, tgt, adapt=True, chunk=chunk, eta=0.1),
            }
            for e in etas:
                runs[f"adapt_e{e}"] = A.stream_nll(feat, base, tgt, adapt=True, chunk=chunk, eta=e)
            for arm, (nll, _) in runs.items():
                acc[bk][arm].append(per_segment_bpc(nll, spans)[L2])
                c, b = recovery_curve(nll[max(la - 1, 0):lb - 1], bin_bytes=max(250, seg // 20))
                curves[bk].setdefault(arm, []).append(b)
            fl2 = runs["frozen"][0][max(la - 1, 0):lb - 1]     # FROZEN L2 nll (in-context probe)
            t = max(1, len(fl2) // 3)
            probe[bk]["early"].append(A.bpc(fl2[:t])); probe[bk]["late"].append(A.bpc(fl2[-t:]))
    return acc, curves, arms, probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pap_msgpack", required=True); ap.add_argument("--pap_meta", required=True)
    ap.add_argument("--pap_inp_msgpack", default=""); ap.add_argument("--pap_inp_meta", default="")  # error-feedback OFF control
    ap.add_argument("--s5_msgpack", required=True); ap.add_argument("--s5_meta", required=True)
    ap.add_argument("--eng", required=True); ap.add_argument("--l2_fr", required=True); ap.add_argument("--l2_ru", required=True)
    ap.add_argument("--seg", type=int, default=20000); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=64); ap.add_argument("--etas", default="0.03,0.1,0.3")
    ap.add_argument("--eng_base", type=int, default=95_000_000); ap.add_argument("--out", default="")
    args = ap.parse_args()
    etas = [float(x) for x in args.etas.split(",")]

    print(f"VERIFY  seg={args.seg} n={args.n} chunk={args.chunk} etas={etas}")
    backbones = [("PAP", B.load_backbone(args.pap_msgpack, args.pap_meta)[:2])]
    if args.pap_inp_msgpack:   # error-feedback OFF (eps=u) -- the causal control for in-context adaptation
        backbones.append(("PAPinp", B.load_backbone(args.pap_inp_msgpack, args.pap_inp_meta)[:2]))
    backbones.append(("S5", B.load_backbone(args.s5_msgpack, args.s5_meta)[:2]))
    names = [bk for bk, _ in backbones]

    conds = [("FR", args.l2_fr, False), ("RU", args.l2_ru, False), ("EN-noshift", None, True)]
    allsave = {}
    for label, midpath, mid_is_eng in conds:
        acc, curves, arms, probe = run_condition(backbones, args.eng, midpath, args.seg, args.n,
                                                 args.chunk, etas, label, mid_is_eng, args.eng_base)
        print(f"\n==================== {label}  (L2-segment BPC, mean+/-std over n={args.n}) ====================")
        for bk in names:
            fr = np.array(acc[bk]["frozen"])
            print(f"  {bk}:")
            for arm in arms:
                v = np.array(acc[bk][arm]); d = fr.mean() - v.mean()
                mk = "  <-" if arm.startswith("adapt") else ("  (ctrl)" if arm in ("bias", "random") else "")
                print(f"      {arm:12s} {v.mean():.3f} +/- {v.std():.3f}"
                      + (f"   dvs_frozen={d:+.3f} ({100*d/fr.mean():+.1f}%)" if arm != "frozen" else "") + mk)
            e, l = np.array(probe[bk]["early"]).mean(), np.array(probe[bk]["late"]).mean()
            print(f"      in-context(frozen): L2 early-third {e:.3f} -> late-third {l:.3f} "
                  f"({100*(e-l)/e:+.1f}% within-segment drop = recurrence adapting in-context)")
        # save curves (averaged across samples) for plotting
        for bk in names:
            for arm, lst in curves[bk].items():
                m = min(len(x) for x in lst); allsave[f"{label}_{bk}_{arm}"] = np.mean([x[:m] for x in lst], 0)
    if args.out:
        np.savez(args.out, **allsave); print(f"\n[*] saved curves -> {args.out}")
    print("\nVERIFY_DONE")


if __name__ == "__main__":
    main()
