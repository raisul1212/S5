"""Stage C local check. Parts 1-2 are self-contained (no checkpoint). Part 3 runs the REAL
recovery check if STAGEC_DIR (with pap_err.msgpack/.meta.pkl + eng.txt/fr.txt) is set.

  STAGEC_DIR=/path python paper_v2_ppac/pap_streaming/test_stageC.py
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import jax, jax.numpy as jnp
from paper_v2_ppac.pap_streaming import stream as S, adapt as A, backbone as B

print("=" * 64); print("STAGE-C LOCAL CHECK"); print("=" * 64)

# --- 1. stream builder (pure) ---
st = S.build_stream([("EN", np.arange(5)), ("FR", np.arange(100, 103)), ("EN2", np.arange(5, 8))])
assert list(st["ids"]) == [0, 1, 2, 3, 4, 100, 101, 102, 5, 6, 7]
assert st["bounds"] == [5, 8] and st["spans"] == [("EN", 0, 5), ("FR", 5, 8), ("EN2", 8, 11)]
print("[1] stream builder OK (ids, bounds, spans correct)")

# --- 2. mechanics: fresh tiny PAP backbone; feature shapes + adapt causality + adapt helps ---
from s5.pap_ssm import init_PAPSSM
from s5.seq_model import LMModel
Dt, Pt, NLt, Vt = 64, 16, 2, 256
model = LMModel(ssm=init_PAPSSM(H=Dt, P=Pt, pap_gate=False), d_output=Vt, d_model=Dt, n_layers=NLt,
                activation="gelu", batchnorm=False, prenorm=True, training=False)
ids = np.tile(np.arange(16), 20).astype(np.int64)[:300]      # repetitive -> learnable online
params = model.init(jax.random.PRNGKey(0), jax.nn.one_hot(jnp.zeros((16,), jnp.int32), Vt), jnp.ones((16,)))["params"]
feat, base = B.extract(model, params, ids[:-1])
assert feat.shape == (len(ids) - 1, Dt) and base.shape == (len(ids) - 1, Vt), (feat.shape, base.shape)
print(f"[2a] feature extraction OK: feat {feat.shape}, base {base.shape}")

tgt = ids[1:]
nll_frozen, _ = A.stream_nll(feat, base, tgt, adapt=False, chunk=32)
nll_adapt, dW = A.stream_nll(feat, base, tgt, adapt=True, eta=0.5, chunk=32)
# causality: first chunk identical (adapt hasn't updated yet -> scored with W=0)
assert np.allclose(nll_frozen[:32], nll_adapt[:32], atol=1e-5), "first chunk must match (score-before-update)"
print("[2b] adapt causality OK: first chunk scored with W=0 (score-before-update)")
# on repetitive data, later chunks improve with adaptation
late_frozen = A.bpc(nll_frozen[64:]); late_adapt = A.bpc(nll_adapt[64:])
print(f"[2c] adapt helps on repetitive stream: late-BPC frozen {late_frozen:.3f} -> adapt {late_adapt:.3f}")
assert late_adapt < late_frozen, "adaptation should lower later-chunk BPC on a learnable stream"

# --- 3. REAL recovery check (needs STAGEC_DIR) ---
d = os.environ.get("STAGEC_DIR", "")
if d and os.path.exists(os.path.join(d, "pap_err.msgpack")):
    print(f"\n[3] REAL check (PAP checkpoint + EN->FR->EN), dir={d}")
    seg = int(os.environ.get("SEG", "500"))
    stR = S.eng_l2_eng(os.path.join(d, "eng.txt"), os.path.join(d, "fr.txt"), seg, l2_label="FR")
    mdl, prm, a = B.load_backbone(os.path.join(d, "pap_err.msgpack"), os.path.join(d, "pap_err.meta.pkl"))
    feat, base = B.extract(mdl, prm, stR["ids"][:-1]); tgt = stR["ids"][1:]
    fz, _ = A.stream_nll(feat, base, tgt, adapt=False, chunk=128)
    segfz = {l: A.bpc(fz[a2:min(b2, len(fz))]) for l, a2, b2 in stR["spans"]}
    print(f"    frozen : " + "  ".join(f"{k}={v:.3f}" for k, v in segfz.items())
          + f"   (EN~1.76 trained -> sanity; FR>>EN -> shift is real)")
    for eta in (0.03, 0.1, 0.3):
        ad, dW = A.stream_nll(feat, base, tgt, adapt=True, eta=eta, chunk=128)
        segad = {l: A.bpc(ad[a2:min(b2, len(ad))]) for l, a2, b2 in stR["spans"]}
        impr = 100 * (segfz["FR"] - segad["FR"]) / segfz["FR"]   # +ve = FR improved by adaptation
        print(f"    adapt eta={eta:<4}: " + "  ".join(f"{k}={v:.3f}" for k, v in segad.items())
              + f"   FR {segfz['FR']:.3f}->{segad['FR']:.3f} ({impr:+.0f}%)  meandW={dW[dW>0].mean():.3f}")
else:
    print("\n[3] SKIPPED real check (set STAGEC_DIR to run it)")

print("\nSTAGE-C LOCAL CHECK PASS")
