"""Sanity check for flop_counter_jaxpr.walk_jaxpr on KNOWN cases.

Each test case has an expected FLOP count from first principles.
We compare walker output vs expected, print PASS/FAIL.

Run:
    python test_flop_walker_sanity.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from flop_counter_jaxpr import walk_jaxpr


def test(name, fn, args, expected_dot_flops, tolerance=0.0):
    """Trace fn(*args), walk jaxpr, check dot_general FLOP count."""
    closed = jax.make_jaxpr(fn)(*args)
    counts = walk_jaxpr(closed.jaxpr)
    actual = counts.get("dot_general", 0)
    if expected_dot_flops == 0:
        ok = actual == 0
        pct = 0
    else:
        pct = 100.0 * (actual - expected_dot_flops) / expected_dot_flops
        ok = abs(pct) <= (tolerance * 100)
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}")
    print(f"         expected dot_general: {expected_dot_flops:,}")
    print(f"         actual dot_general:   {actual:,}   (delta {pct:+.1f}%)")
    if not ok:
        print(f"         all counts: {dict((k, v) for k, v in counts.items() if not k.startswith('_') and v > 0)}")
    return ok


print("=" * 70)
print("FLOP walker sanity tests")
print("=" * 70)

# ---- Case 1: Real matvec, no batch ----
# A shape (M=100, N=200) real, b shape (N=200,) real
# Expected: 2 * M * N = 40,000 real FLOPs
def fn1(A, b):
    return A @ b
A = jnp.ones((100, 200))
b = jnp.ones((200,))
test("Real matvec (100, 200) @ (200,)", fn1, (A, b), 2 * 100 * 200)

# ---- Case 2: Real matmul batched ----
# A shape (2048, 100), B shape (100, 200)
# Expected: 2 * 2048 * 100 * 200 = 82M
def fn2(A, B):
    return A @ B
A2 = jnp.ones((2048, 100))
B2 = jnp.ones((100, 200))
test("Real batched matmul (2048, 100) @ (100, 200)",
     fn2, (A2, B2), 2 * 2048 * 100 * 200)

# ---- Case 3: Vmap over L: matvec applied L times ----
# W shape (32, 128) real, x shape (2048, 128) real
# per token: 2 * 32 * 128 = 8192
# L tokens: 2048 * 8192 = 16.8M
def fn3(W, x_seq):
    return jax.vmap(lambda x: W @ x)(x_seq)
W3 = jnp.ones((32, 128))
x3 = jnp.ones((2048, 128))
test("vmap matvec, L=2048, (32, 128) @ (128,)",
     fn3, (W3, x3), 2 * 2048 * 32 * 128)

# ---- Case 4: Complex-complex matmul ----
# C shape (100, 64) complex, h shape (64,) complex
# Expected: 8 * 100 * 64 = 51,200 FLOPs (complex mul-add fused)
def fn4(C, h):
    return C @ h
C4 = jnp.ones((100, 64), dtype=jnp.complex64)
h4 = jnp.ones((64,), dtype=jnp.complex64)
test("Complex-complex matvec (100, 64) @ (64,)",
     fn4, (C4, h4), 8 * 100 * 64)

# ---- Case 5: Complex-real matmul (B_bar @ x_real) ----
# B complex (32, 128), x real (128,) -> complex output size 32
# Real formula: 4 * M * N per matvec (2 real muls for real part + 2 for imag)
# But my walker uses 8 if EITHER is complex.  We check what it actually reports.
def fn5(B, x):
    return B @ x
B5 = jnp.ones((32, 128), dtype=jnp.complex64)
x5 = jnp.ones((128,))
# Note: my walker will report 8 * 32 * 128 = 32,768 (overcount)
# Physical minimum: 4 * 32 * 128 = 16,384
test("Complex-real matvec (32, 128) complex @ (128,) real",
     fn5, (B5, x5), 8 * 32 * 128)  # expecting walker's 8x behavior

# ---- Case 6: (C @ h).real -- does XLA drop imag? ----
# We'll trace and inspect the primitives
def fn6(C, h):
    return (C @ h).real
C6 = jnp.ones((128, 64), dtype=jnp.complex64)
h6 = jnp.ones((64,), dtype=jnp.complex64)
closed6 = jax.make_jaxpr(fn6)(C6, h6)
counts6 = walk_jaxpr(closed6.jaxpr)
print(f"\n  (C @ h).real trace:")
print(f"    primitives: {dict((k, v) for k, v in counts6.items() if v > 0 and not k.startswith('_'))}")
print(f"    walker reports {counts6.get('dot_general', 0):,} dot_general FLOPs")
print(f"    full complex matmul: {8 * 128 * 64:,} FLOPs")
print(f"    real-only optimization: {2 * 128 * 64:,} FLOPs (theoretical minimum)")

# ---- Case 7: Simple associative_scan ----
# scan (h, x) -> (Lambda*h + x, out) over L=100 steps, state=8
# Per step body: 1 complex mul + 1 complex add = 6 + 2 = 8 real FLOPs
# But associative_scan does log(L) tree work per element; total combines = 2*L
# Per combine: 14 real FLOPs (2 complex muls + 1 complex add for tuple combine)
# Total: 2 * L * P * 14 = 2 * 100 * 8 * 14 = 22,400
def fn7(Lambda, Bu):
    def combine(qi, qj):
        Ai, bi = qi
        Aj, bj = qj
        return Aj * Ai, Aj * bi + bj
    _, out = jax.lax.associative_scan(combine, (Lambda, Bu))
    return out
Lambda7 = jnp.ones((100, 8), dtype=jnp.complex64)
Bu7 = jnp.ones((100, 8), dtype=jnp.complex64)
closed7 = jax.make_jaxpr(fn7)(Lambda7, Bu7)
counts7 = walk_jaxpr(closed7.jaxpr)
print(f"\n  associative_scan (L=100, P=8, complex) trace:")
print(f"    primitives: {dict((k, v) for k, v in counts7.items() if v > 0 and not k.startswith('_'))}")
print(f"    total FLOPs: {sum(v for k, v in counts7.items() if not k.startswith('_')):,}")
print(f"    naive expected (2*L*P*14): {2 * 100 * 8 * 14:,}")

# ---- Case 8: Ratio invariance -- iso-work should give iso-FLOPs ----
# Test that two DIFFERENT arrangements of same total work give same FLOPs
# Arrangement A: one (128, 32) matmul
# Arrangement B: two (128, 16) matmuls
# Both should be 2 * 128 * 32 = 8192 total dot_general FLOPs
def fn8a(A, b):
    return A @ b
def fn8b(A1, A2, b1, b2):
    return A1 @ b1 + A2 @ b2
A8a = jnp.ones((128, 32))
b8a = jnp.ones((32,))
A8b1 = jnp.ones((128, 16))
A8b2 = jnp.ones((128, 16))
b8b1 = jnp.ones((16,))
b8b2 = jnp.ones((16,))

r_a = walk_jaxpr(jax.make_jaxpr(fn8a)(A8a, b8a).jaxpr).get("dot_general", 0)
r_b = walk_jaxpr(jax.make_jaxpr(fn8b)(A8b1, A8b2, b8b1, b8b2).jaxpr).get("dot_general", 0)
print(f"\n  Iso-work test:")
print(f"    A: one (128, 32) matmul     -> {r_a:,} FLOPs")
print(f"    B: two (128, 16) matmuls    -> {r_b:,} FLOPs")
print(f"    Equal? {'YES' if r_a == r_b else 'NO'}")

print()
print("=" * 70)
print("Sanity check done.")
print("=" * 70)
