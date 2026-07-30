"""Direct-instrumentation PPAC for the Mambino paper — v3.

Consumes:
  - workload_{cfg}_gemms.yaml     -- JAXPR-extracted per-GEMM records, hard-invariant checked
  - workload_{cfg}_elemwise.yaml  -- JAXPR-extracted elementwise counts by energy class
  - workload_{cfg}_manifest.yaml  -- checkpoint provenance + args (n_layers, L, bidir, etc.)
  - out_accelergy_{ax}x{ay}/ERT_summary.yaml -- per-access energies (CACTI 7 + NeuroSim)

Produces, for each (config × array_size):
  cycles, energy_pJ, per-component energy breakdown, power_mW, latency_ms,
  EDP, acc/mJ, acc*throughput/W. Every number derivable by hand from inputs.

Chip: 22 nm INT8 WS systolic, 256 KB weight SRAM + 128 KB activation SRAM + 64 KB
state SRAM. Buffer sizes held FIXED across the array sweep (only PE count changes).

WS execution model:
  For GEMM A[M×K] · B[K×N] on an (ax × ay) array:
    C_sp = largest divisor of K with C_sp <= ay   (spatial fanout of K on array_y)
    K_sp = largest divisor of N with K_sp <= ax   (spatial fanout of N on array_x)
    C_tp = K / C_sp,   K_tp = N / K_sp             (temporal tile factors)
    cycles = M * C_tp * K_tp                       (M is temporal activation stream)
    util   = (C_sp * K_sp) / (ax * ay)

Per-GEMM memory traffic (bytes; INT8 so bytes == elements):
  weight_sram_bytes_read  = K * N               (each weight loaded once per GEMM)
  activation_sram_bytes_read  = M * K * K_tp    (streamed once per K-tile of N)
  activation_sram_bytes_write = M * N           (outputs written once)
  weights_spad_writes = C_sp * K_sp * C_tp * K_tp = K * N  (each PE loaded per weight tile)
  weights_spad_reads  = total_real_MACs         (each PE reads its weight every MAC)
  psum_spad reads+writes = total_real_MACs each (accumulate per MAC)

State SRAM traffic (SSM state):
  reads_per_infer  = L * n_layers * n_directions * n_traj * ceil(2*P / line_bytes)
  writes = same. n_traj = 2 for Pure S5, 3 for Mambino (main + fwd + bwd + predictor fwd).

dtype scaling for complex GEMMs (per JAXPR-faithful convention):
  complex×real   -> per_mac_flops=4, dtype_scale=2  (2 real MACs / logical MAC)
  complex×complex -> per_mac_flops=8, dtype_scale=4 (4 real MACs / logical MAC)
Cycles and MAC-array energy scale by dtype_scale; SRAM access counts do NOT (the
memory sees INT8 bytes regardless of algebraic interpretation).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import yaml

# --- Chip params (must match build_chip_specs.py) --------------------------
CHIP = dict(
    freq_hz=1e9,
    line_bytes=8,                # SRAM line = 64 bits = 8 bytes
    weight_sram_bytes=256 * 1024,
    activation_sram_bytes=384 * 1024,   # default; per-config override via CONFIG_TIER
    state_sram_bytes=64 * 1024,
)

# Per-config chip tier: activation SRAM sized to fit that config's workload.
# Tier index = suffix on out_accelergy_{tier}_{ax}x{ay}/ dir names.
CONFIG_TIER = {
    "config4":  "320KB",   # 106K NoGLU (max footprint 258 KB)
    "config5":  "320KB",   # 106K NoGLU (max footprint 260 KB)
    "corner2":  "384KB",   # 188K low-rank half_glu2 (max footprint 336 KB)
    "corner3p": "384KB",   # 188K low-rank half_glu2 (max footprint 336 KB)
    "corner1":  "512KB",   # 188K full-rank dense half_glu2 (max footprint 512 KB)
}
TIER_TO_BYTES = {"320KB": 320*1024, "384KB": 384*1024, "512KB": 512*1024}

ARRAY_SIZES = [(8, 8), (16, 16), (32, 32), (64, 64)]

ACC = {"config4": 0.5993, "config5": 0.5917,
       "corner1": 0.6089, "corner2": 0.5991, "corner3p": 0.6138}

LABEL = {"config4": "Config 4", "config5": "Config 5",
         "corner1": "Corner 1", "corner2": "Corner 2", "corner3p": "Corner 3'"}
FAMILY = {"config4": "Mambino", "config5": "Pure S5",
          "corner1": "Pure S5", "corner2": "Pure S5", "corner3p": "Mambino"}
BAND = {"config4": "106K", "config5": "106K",
        "corner1": "188K", "corner2": "188K", "corner3p": "188K"}

# Trajectories per inference (see master doc §3):
#   Pure S5: 2 (main forward + main backward)
#   Mambino: 3 (main forward + main backward + predictor forward)
N_TRAJ = {"config4": 3, "config5": 2, "corner1": 2, "corner2": 2, "corner3p": 3}

# Elementwise op-class energy cost model (pJ per scalar op).
# Anchored on the mac_random energy from the ERT so it scales with tech node.
ELEMWISE_COST_UNITS_OF_MAC = {
    "trivial": 1.0,          # add/sub -- one adder cycle
    "moderate": 3.0,         # mul/div/pow -- small ALU sequence
    "transcendental": 10.0,  # exp/log/tanh/sig/rsqrt -- LUT + interpolation
    "reduction": 2.0,        # reduce_sum -- chain of adds, memory-bound
    "structural": 0.0,       # reshape/broadcast -- pure metadata, no compute
}


# --- Helpers --------------------------------------------------------------
def gemm_is_complex_output(dtype: str) -> bool:
    """True when the GEMM emits a complex-valued output (both operands complex)."""
    return dtype == "complex_complex"


def factor_split(dim: int, spatial_max: int) -> tuple[int, int]:
    """Largest divisor of `dim` that is <= spatial_max; return (spatial, temporal)."""
    if dim <= spatial_max:
        return dim, 1
    best = 1
    for f in range(1, spatial_max + 1):
        if dim % f == 0 and f > best:
            best = f
    return best, dim // best


def parse_ert(ert_path: Path) -> dict[str, dict[str, float]]:
    """Return {short_component_name: {action_name: pJ_per_access}}."""
    ert = yaml.safe_load(open(ert_path))["ERT_summary"]["table_summary"]
    out: dict[str, dict[str, float]] = {}
    for c in ert:
        short = c["name"].split(".")[-1]
        # De-duplicate: multiple PE[N] entries all have same per-access energy;
        # keep the first (they are identical anyway).
        if short not in out:
            out[short] = {a["name"]: float(a["energy"]) for a in c["actions"]}
    return out


# --- Per-GEMM PPAC --------------------------------------------------------
def gemm_ppac(g: dict, array_x: int, array_y: int, ert: dict) -> dict:
    """Compute cycles + energy for a single GEMM instance (one repeat)."""
    M, N, K = g["M"], g["N"], g["K"]
    dtype_scale = g["per_mac_flops"] // 2   # 1 (real), 2 (cxr), 4 (cxc)

    C_sp, C_tp = factor_split(K, array_y)
    K_sp, K_tp = factor_split(N, array_x)
    util_pct = 100.0 * (C_sp * K_sp) / (array_x * array_y)
    cycles = M * C_tp * K_tp * dtype_scale     # dtype_scale = extra passes for complex

    # Memory bytes moved for one instance.
    # Weights + activations: `dtype_scale` extra passes for complex GEMMs.
    # Outputs: complex×real emits a REAL output (dtype_scale=2 passes,
    # 1 real byte written), complex×complex emits a COMPLEX output
    # (dtype_scale=4 passes, 2 real bytes written -> output_scale = dtype_scale/2).
    output_scale = 2 if gemm_is_complex_output(g["dtype"]) else 1
    weight_bytes_read  = K * N * dtype_scale
    act_read_bytes  = M * K * K_tp * dtype_scale   # re-streamed per N-tile
    act_write_bytes = M * N * output_scale
    real_macs = M * N * K * dtype_scale

    # Buffer capacity check against this config's chip tier.
    # (Weight SRAM stays at 256 KB across all tiers; activation SRAM varies.)
    act_budget_bytes = TIER_TO_BYTES.get(getattr(gemm_ppac, "_tier", "384KB"), 384*1024)
    capacity_ok = (
        K * N <= CHIP["weight_sram_bytes"]
        and (M * K + M * N) <= act_budget_bytes
    )

    def lines(b): return math.ceil(b / CHIP["line_bytes"])

    e_weight_sram = lines(weight_bytes_read) * ert["weight_sram"]["read"]
    e_act_sram    = (lines(act_read_bytes)  * ert["activation_sram"]["read"]
                   + lines(act_write_bytes) * ert["activation_sram"]["write"])
    e_weights_spad = real_macs * ert["weights_spad"]["read"]
    e_psum_spad    = real_macs * (ert["psum_spad"]["read"] + ert["psum_spad"]["update"])
    e_mac          = real_macs * ert["mac"]["compute"]

    total_pJ = e_weight_sram + e_act_sram + e_weights_spad + e_psum_spad + e_mac
    return dict(
        M=M, N=N, K=K, dtype=g["dtype"],
        cycles=cycles, util_pct=round(util_pct, 2),
        real_macs=real_macs,
        capacity_ok=capacity_ok,
        e_mac_pJ=round(e_mac, 3),
        e_weight_sram_pJ=round(e_weight_sram, 3),
        e_act_sram_pJ=round(e_act_sram, 3),
        e_weights_spad_pJ=round(e_weights_spad, 3),
        e_psum_spad_pJ=round(e_psum_spad, 3),
        total_pJ=round(total_pJ, 3),
    )


# --- Elementwise PPAC -----------------------------------------------------
def elemwise_energy(elem_yaml: dict, ert: dict) -> tuple[float, dict[str, float]]:
    """Return (total_pJ, per_class_pJ) for the config's elementwise workload."""
    mac_pJ = ert["mac"]["compute"]
    total = 0.0
    per_class: dict[str, float] = {}
    for cls, tot in elem_yaml["energy_class_totals"].items():
        ops = tot["total_scalar_ops"]
        e = ops * mac_pJ * ELEMWISE_COST_UNITS_OF_MAC.get(cls, 1.0)
        per_class[cls] = round(e, 3)
        total += e
    return total, per_class


# --- State-SRAM PPAC ------------------------------------------------------
def state_sram_energy(manifest: dict, config: str, ert: dict) -> tuple[float, int, int]:
    """SSM state read/write per inference, INT8.
    Reads = L * n_layers * n_directions * n_traj * ceil(2*P / line_bytes) lines.
    Writes = same.
    """
    args = manifest["args"]
    L = 2048  # LRA-ListOps sequence length
    n_layers = args["n_layers"]
    n_dir = 2 if args["bidirectional"] else 1
    P = args["ssm_size_base"] // 2   # conj_sym halves it
    n_traj = N_TRAJ[config]
    state_bytes_per_step = 2 * P     # 2 real INT8 bytes per complex state entry
    lines_per_step = math.ceil(state_bytes_per_step / CHIP["line_bytes"])
    total_reads = L * n_layers * n_dir * n_traj * lines_per_step
    total_writes = total_reads   # same access pattern
    e = (total_reads * ert["state_sram"]["read"]
       + total_writes * ert["state_sram"]["write"])
    return e, total_reads, total_writes


# --- Per-config PPAC ------------------------------------------------------
def config_ppac(config: str, workload_dir: Path, chip_dir: Path,
                array_x: int, array_y: int) -> dict:
    gemms = yaml.safe_load(open(workload_dir / f"workload_{config}_gemms.yaml"))["gemms"]
    elemw = yaml.safe_load(open(workload_dir / f"workload_{config}_elemwise.yaml"))
    manif = yaml.safe_load(open(workload_dir / f"workload_{config}_manifest.yaml"))
    tier = CONFIG_TIER[config]
    ert = parse_ert(chip_dir / f"out_accelergy_{tier}_{array_x}x{array_y}" / "ERT_summary.yaml")
    gemm_ppac._tier = tier  # noqa: set for capacity check to read config's tier

    per_gemm_results = []
    tot_cycles = 0
    tot_pJ = 0.0
    comp_pJ: dict[str, float] = {}
    tot_real_macs = 0

    for g in gemms:
        one = gemm_ppac(g, array_x, array_y, ert)
        rep = g["repeat"]
        tot_cycles += one["cycles"] * rep
        tot_real_macs += one["real_macs"] * rep
        tot_pJ += one["total_pJ"] * rep
        for k in ("e_mac_pJ", "e_weight_sram_pJ", "e_act_sram_pJ",
                  "e_weights_spad_pJ", "e_psum_spad_pJ"):
            comp = k.replace("e_", "").replace("_pJ", "")
            comp_pJ[comp] = comp_pJ.get(comp, 0.0) + one[k] * rep
        per_gemm_results.append({**one, "repeat": rep,
                                 "total_pJ_with_repeat": round(one["total_pJ"] * rep, 3)})

    e_elem, per_class = elemwise_energy(elemw, ert)
    tot_pJ += e_elem
    for cls, e in per_class.items():
        comp_pJ[f"elemwise_{cls}"] = e

    e_state, state_reads, state_writes = state_sram_energy(manif, config, ert)
    tot_pJ += e_state
    comp_pJ["state_sram"] = round(e_state, 3)

    # Leakage: per-cycle × total cycles. The SRAM leaks are chip-level (single instance),
    # but the mac + weights_spad + psum_spad leaks are PE-level (n_pe instances).
    # parse_ert dedupes the PE[0..N] entries to a single per-PE value, so multiply
    # those three components by n_pe.
    n_pe = array_x * array_y
    per_pe_leak = (ert["mac"].get("leak", 0.0)
                 + ert["weights_spad"].get("leak", 0.0)
                 + ert["psum_spad"].get("leak", 0.0))
    chip_leak = (ert["weight_sram"].get("leak", 0.0)
               + ert["activation_sram"].get("leak", 0.0)
               + ert["state_sram"].get("leak", 0.0))
    leak_pJ_per_cyc = chip_leak + per_pe_leak * n_pe
    e_leak = leak_pJ_per_cyc * tot_cycles
    tot_pJ += e_leak
    comp_pJ["leakage"] = round(e_leak, 3)

    latency_s = tot_cycles / CHIP["freq_hz"]
    energy_J = tot_pJ / 1e12
    power_W = energy_J / latency_s if latency_s > 0 else 0
    tput_ips = 1.0 / latency_s if latency_s > 0 else 0
    acc = ACC[config]
    edp_J_s = energy_J * latency_s

    return dict(
        config=config, label=LABEL[config], family=FAMILY[config], params_band=BAND[config],
        chip_tier=tier,
        accuracy=acc, array_x=array_x, array_y=array_y, n_pe=array_x * array_y,
        total_cycles=tot_cycles,
        latency_ms=round(latency_s * 1e3, 4),
        energy_mJ=round(tot_pJ / 1e9, 5),
        power_mW=round(power_W * 1e3, 3),
        throughput_ips=round(tput_ips, 3),
        total_real_macs=tot_real_macs,
        pj_per_real_mac=round(tot_pJ / tot_real_macs, 4) if tot_real_macs else 0,
        acc_per_mJ=round(acc / (tot_pJ / 1e9), 5) if tot_pJ > 0 else 0,
        acc_x_throughput_per_W=round(acc * tput_ips / power_W, 3) if power_W > 0 else 0,
        edp_mJ_ms=round(edp_J_s * 1e3 * 1e3, 5),
        state_sram_reads=state_reads, state_sram_writes=state_writes,
        per_component_pJ=comp_pJ,
        per_gemm=per_gemm_results,
    )


# --- Main -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads_dir",
                    default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workloads")))
    ap.add_argument("--chip_dir",
                    default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    workloads = Path(args.workloads_dir)
    chip = Path(args.chip_dir)
    configs = ["config4", "config5", "corner1", "corner2", "corner3p"]

    per_config: dict[str, dict[str, dict]] = {}
    sweep_rows = []
    per_gemm_flat: dict[str, dict] = {}

    for cfg in configs:
        per_config[cfg] = {}
        for ax, ay in ARRAY_SIZES:
            r = config_ppac(cfg, workloads, chip, ax, ay)
            key = f"{ax}x{ay}"
            per_config[cfg][key] = r
            row = {k: v for k, v in r.items() if k not in ("per_gemm", "per_component_pJ")}
            row["per_component_pJ"] = r["per_component_pJ"]
            sweep_rows.append(row)
            for gg in r["per_gemm"]:
                per_gemm_flat[f"{cfg}__M{gg['M']}_N{gg['N']}_K{gg['K']}_{gg['dtype']}__{key}"] = gg

    # Write outputs
    (chip / "per_config_ppac_v3.json").write_text(
        json.dumps(per_config, indent=2, default=str))
    (chip / "per_gemm_ppac_v3.json").write_text(
        json.dumps(per_gemm_flat, indent=2, default=str))

    hdr = ["config", "label", "family", "params_band", "accuracy",
           "array_x", "array_y", "n_pe",
           "total_cycles", "latency_ms", "energy_mJ", "power_mW",
           "throughput_ips", "total_real_macs", "pj_per_real_mac",
           "acc_per_mJ", "acc_x_throughput_per_W", "edp_mJ_ms",
           "state_sram_reads", "state_sram_writes"]
    with open(chip / "sweep_summary_v3.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r in sweep_rows:
            w.writerow([r[k] for k in hdr])
    print(f"[direct] wrote per_config_ppac_v3.json, per_gemm_ppac_v3.json, sweep_summary_v3.csv")

    # Report
    print(f"\n{'='*128}")
    print(f"Direct-instrumentation PPAC — {len(sweep_rows)} rows (5 configs × {len(ARRAY_SIZES)} array sizes)")
    print(f"{'='*128}")
    print(f"{'Config':<11}{'Array':<8}{'Acc':>8}{'Cycles':>12}{'Latency':>10}"
          f"{'Energy':>10}{'Power':>10}{'TPS':>9}{'acc/mJ':>10}{'pJ/MAC':>10}{'EDP':>10}")
    print("-"*128)
    for r in sweep_rows:
        print(f"{r['config']:<11}{r['array_x']}x{r['array_y']:<6}"
              f"{r['accuracy']:>8.4f}"
              f"{r['total_cycles']:>12,}"
              f"{r['latency_ms']:>8.3f}ms"
              f"{r['energy_mJ']:>8.3f}mJ"
              f"{r['power_mW']:>8.1f}mW"
              f"{r['throughput_ips']:>9.1f}"
              f"{r['acc_per_mJ']:>10.4f}"
              f"{r['pj_per_real_mac']:>10.3f}"
              f"{r['edp_mJ_ms']:>10.4f}")


if __name__ == "__main__":
    main()
