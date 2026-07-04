"""Chip PPAC (Performance, Power, Area, Cost) estimation for the 5-config
Mambino paper.  Uses PUBLISHED METHODOLOGIES and EQUATIONS, not fitted
numbers.

METHODOLOGY -- see published frameworks below.

DIGITAL implementation methodology
==================================
Uses the Accelergy energy model (Wu, Sze, Emer, RSP 2019 -- Accelergy:
An Architecture-Level Energy Estimation Methodology for Accelerator
Designs, https://ieeexplore.ieee.org/document/8942149):
  E_total = Sum over components C: N_actions(C) * E_per_action(C)

Component actions:
  MAC: 1 action per multiply-accumulate
  SRAM: N actions per read/write

Latency uses the Timeloop model (Parashar et al. ISPASS 2019,
https://ieeexplore.ieee.org/document/8695666) simplified as:
  Latency = MAC_total / (MAC_array_width * clock_freq)

MIXED-SIGNAL implementation methodology
=======================================
Uses the NeuroSim framework methodology (Chen, Peng, Yu -- NeuroSim:
A Circuit-Level Macro Model for Benchmarking Neuro-Inspired
Architectures in Online Learning, IEDM 2018,
https://ieeexplore.ieee.org/document/8614591):
  E_analog_crossbar = N_ops_MAC * E_analog_MAC_per_op
  E_ADC_periphery   = N_ADC_samples * (FoM_ADC * 2^bits)
  E_DAC_periphery   = N_DAC_samples * (FoM_DAC * 2^bits)
  E_digital_periphery = N_digital_MACs * E_INT8_MAC + SRAM access energy

  Latency = max(t_crossbar_settle, t_ADC * n_columns, t_digital_periphery)
            * n_pipeline_stages

  Area = A_crossbar + A_ADC * n_ADCs + A_DAC * n_DACs +
         A_digital_periphery + A_SRAM

CONSTANTS -- all citable
========================
INT8 MAC (digital, 22nm) = 0.4 pJ
  Source: Horowitz, ISSCC 2014, "Computing's Energy Problem"
    Table 1 gives 45nm INT8 = 0.2 pJ.  Dennard-like scaling to 22nm: 0.4 pJ.
    https://ieeexplore.ieee.org/document/6757323

SRAM 8-bit read (22nm cached) = 0.05 pJ
  Source: Chen et al. Eyeriss ISSCC 2016, Table VII, 65nm scaled.
    https://ieeexplore.ieee.org/document/7738524

SRAM cell area (22nm 6T) = 0.081 um^2/bit
  Source: Wong et al. Sci Rep 2015.
    https://www.nature.com/articles/srep11250

SAR ADC FoM (best 22nm) = 5 fJ/conversion-step
  Source: Murmann ADC Survey, updated annually.
    https://web.stanford.edu/~murmann/adcsurvey.html
  Energy per sample = FoM * 2^bits.  For 8-bit: 5 fJ * 256 = 1.28 pJ.
  Area = 100 * 2^bits um^2 (SAR scaling, area ~ capacitor DAC).

Analog crossbar MAC (gain-cell 22nm) = 10 fJ/MAC
  Source: Marinella et al. JETC 2018, middle of 5-50 fJ published range.
    https://dl.acm.org/doi/10.1145/3230639
  Ni et al. VLSI 2019 gain-cell area: ~25 um^2/cell at 22nm.
    https://ieeexplore.ieee.org/document/8776506

Wafer cost (22nm TSMC ULP) = ~$3,500 (public 2023 estimate)
Yield (22nm mature process) = ~80%
"""
import argparse
from dataclasses import dataclass
import json


# =========== 22nm technology constants (all citable) ===========
# ENERGY (pJ per event) -- Sze 2020 "Efficient Processing of DNNs" breakdown:
#   Per INT8 MAC operation, the FULL cost includes:
#     - compute (mul + accumulate): 0.2 pJ  [Horowitz 2014 scaled]
#     - weight fetch from L1 SRAM:  0.5 pJ  [Chen 2016 Eyeriss]
#     - activation fetch from L1:   0.5 pJ  [Chen 2016]
#     - accumulate to register:     0.1 pJ  [Horowitz 2014]
#   Total per-MAC with memory access: ~1.3 pJ
#
# We track compute and memory access SEPARATELY (Accelergy methodology).
E_INT8_MAC_COMPUTE = 0.2  # compute only (mul + acc), Horowitz 2014 scaled 45nm->22nm
E_SRAM_READ_8b = 0.5      # per 8-bit L1 read, Chen 2016 Eyeriss (32KB L1)
E_SRAM_WRITE_8b = 1.0     # per 8-bit L1 write (typically 2x read)
E_REG_ACCUM = 0.1         # accumulator register access, Horowitz 2014
E_ANALOG_MAC = 0.010      # Marinella 2018 (JETC), gain-cell middle estimate
FoM_ADC = 0.005           # Murmann survey, 5 fJ/conv-step for best SAR
FoM_DAC = 0.002           # DAC typically 2-3x more efficient than ADC

# AREA (um^2)
A_INT8_MAC = 200          # Yin 2020 (JSSC), compact INT8 PE
A_SRAM_BIT = 0.081        # Wong 2015 (Sci Rep), 22nm 6T SRAM
A_ANALOG_CELL = 25        # Ni 2019 (VLSI), gain-cell 22nm
A_ADC_UNIT = 100          # SAR scaling: A ~ 100 * 2^bits um^2
A_DAC_UNIT = 50

# LEAKAGE (mW / mm^2)
LEAKAGE_mW_per_mm2 = 5    # Horowitz 2014, typical 22nm logic

# SYSTEM
CLOCK_GHZ = 1.0
MAC_ARRAY_WIDTH = 1024    # parallel MACs in digital array

# COST
WAFER_COST_USD = 3500     # TSMC 22nm ULP 2023
DIE_AREA_mm2_TARGET = 25  # 5x5 mm target die
DIES_PER_WAFER = 400
YIELD = 0.80

# MODEL
L_SEQ = 2048
H = 128
N_LAYERS = 8


# =========== Config specs (from actual trained models) ===========
@dataclass
class Config:
    name: str
    arch: str
    activation: str
    ssm_size_base: int
    glu_rank: int
    params: int
    train_acc: float          # best test accuracy from training
    # JAXPR-verified FLOP counts (from flop_counter_jaxpr.py with 3-way
    # dtype-aware cost model).  Set to 0 if not yet measured.
    jaxpr_total_flops: int = 0
    jaxpr_dot_general_flops: int = 0  # matmul FLOPs (SSM crossbars + digital gate)

    @property
    def local_P(self):
        return self.ssm_size_base

    @property
    def has_gate(self):
        return self.activation in ("half_glu2", "half_glu1", "full_glu")


# ============================================================
# JAXPR-verified FLOP counts (2026-07-03, batch=1 per-inference)
# From flop_counter_jaxpr.py with corrected 3-way dtype handling:
#   real-real       -> 2 flops/MAC
#   complex-real    -> 4 flops/MAC (B, B_s, W_eps in our SSMs)
#   complex-complex -> 8 flops/MAC (C_tilde @ h)
# Sanity-checked against known matmul cases (all pass with delta=0%).
# ============================================================
CONFIGS = [
    # Corner configs (half_glu2) -- JAXPR counts pending, using estimates
    Config("Corner_1_PureS5_hg2",  "pure_s5", "half_glu2", 16, 0,  188490, 0.6155,
           jaxpr_total_flops=0, jaxpr_dot_general_flops=0),
    Config("Corner_3p_Mambino_hg2","mambino", "half_glu2", 16, 40, 188682, 0.6140,
           jaxpr_total_flops=0, jaxpr_dot_general_flops=0),
    # Gelu configs -- JAXPR-VERIFIED counts (job 11187683)
    Config("Config_3_PureS5_gelu", "pure_s5", "gelu",      16, 0,  56394,  0.5910,
           jaxpr_total_flops=399_216_284,  jaxpr_dot_general_flops=346_032_640),
    Config("Config_4_Mambino_gelu","mambino", "gelu",      16, 0,  105738, 0.6055,
           jaxpr_total_flops=682_095_196,  jaxpr_dot_general_flops=614_468_096),
    Config("Config_5_PureS5_P16",  "pure_s5", "gelu",      32, 0,  105738, 0.5830,
           jaxpr_total_flops=745_208_540,  jaxpr_dot_general_flops=681_576_960),
]


# =========== Op counting per config ===========
def count_ops(cfg: Config):
    """Return per-inference op counts.  All per single input sequence
    of L_SEQ tokens through N_LAYERS layers."""
    local_P = cfg.local_P
    L = L_SEQ

    # Per-layer, per-token analog crossbar MAC counts (SSM matmuls)
    # B: H x local_P complex -> H*local_P*2 real MACs per token
    # C: bidirectional H x local_P complex -> 2 * H*local_P*2 = 4*H*local_P
    # D: H per token
    ssm_MACs_per_layer_per_token = (
        H * local_P * 2       # B
        + 4 * H * local_P     # C (bidir)
        + H                   # D
    )

    if cfg.arch == "mambino":
        # Predictor B_s, C_s (each H*local_P complex)
        # W_eps: local_P x H
        mambino_extra_per_layer_per_token = (
            H * local_P * 2   # B_s
            + H * local_P * 2 # C_s (unidir predictor readout)
            + local_P * H     # W_eps
        )
    else:
        mambino_extra_per_layer_per_token = 0

    analog_MACs_per_token_per_layer = (
        ssm_MACs_per_layer_per_token + mambino_extra_per_layer_per_token
    )
    analog_MACs_total = analog_MACs_per_token_per_layer * L * N_LAYERS

    # Per-layer, per-token DIGITAL MAC counts (gate + BN + activation)
    if cfg.has_gate:
        if cfg.glu_rank > 0:
            gate_MACs = 2 * H * cfg.glu_rank
        else:
            gate_MACs = H * H
        multiply_ops = H
    else:
        gate_MACs = 0
        multiply_ops = 0
    bn_ops = 2 * H            # scale + shift affine
    activation_ops = H        # gelu / sigmoid pointwise

    digital_MACs_per_token_per_layer = (
        gate_MACs + multiply_ops + bn_ops + activation_ops
    )
    # Encoder + decoder MACs (once per inference)
    encoder_MACs = 20 * H * L  # ListOps vocab -> embedding
    decoder_MACs = H * 10      # H -> classes

    digital_MACs_total = (
        digital_MACs_per_token_per_layer * L * N_LAYERS
        + encoder_MACs + decoder_MACs
    )

    # SRAM: weights + inter-layer activations + intra-layer state trajectory
    weight_bytes = cfg.params
    # Inter-layer activation flow (H per token per layer, INT8)
    activation_bytes = 2 * H * L * N_LAYERS
    # Intra-layer state trajectory (materialized during parallel scan)
    # Complex values = 2 real bytes per element at INT8.
    if cfg.arch == "mambino":
        # Main bidir + predictor unidir
        state_bytes_per_layer = 2 * L * local_P * 2 + L * local_P * 2
    else:
        # Main bidir only
        state_bytes_per_layer = 2 * L * local_P * 2
    state_bytes_total = state_bytes_per_layer * N_LAYERS
    sram_bytes_total = weight_bytes + activation_bytes + state_bytes_total

    # ADC/DAC crossings (assuming crossings_every=1)
    # One DAC-in + one ADC-out per layer per token; each transfers H samples
    adc_samples = N_LAYERS * L * H
    dac_samples = N_LAYERS * L * H
    n_adc_units = N_LAYERS    # one per layer
    n_dac_units = N_LAYERS

    # Digital-side param bytes (for mixed-signal SRAM sizing)
    if cfg.has_gate:
        gate_params_per_layer = (
            2 * H * cfg.glu_rank if cfg.glu_rank > 0 else H * H
        ) + H  # bias
    else:
        gate_params_per_layer = 0
    bn_params_per_layer = 2 * H
    digital_side_params = (
        N_LAYERS * (gate_params_per_layer + bn_params_per_layer)
        + 20 * H + H     # encoder
        + H * 10 + 10    # decoder
    )
    analog_side_params = cfg.params - digital_side_params
    if analog_side_params < 0:
        analog_side_params = cfg.params

    # Override with JAXPR-verified counts if available.
    # JAXPR FLOPs -> MACs conversion: 1 MAC = 2 FLOPs (real-real convention).
    # For our SSMs, effective MAC count = FLOPs / 2 (average across dtype mix).
    # This treats complex-real as 2 effective MACs (matches 4 flops / 2 = 2 MACs)
    # and complex-complex as 4 effective MACs (8 flops / 2 = 4 MACs).
    if cfg.jaxpr_total_flops > 0:
        # Use JAXPR-authoritative counts
        total_MACs_verified = cfg.jaxpr_total_flops // 2
        analog_MACs_verified = cfg.jaxpr_dot_general_flops // 2
        digital_MACs_verified = (cfg.jaxpr_total_flops - cfg.jaxpr_dot_general_flops) // 2
        return {
            "analog_MACs": analog_MACs_verified,
            "digital_MACs": digital_MACs_verified,
            "sram_bytes": sram_bytes_total,
            "weight_bytes": weight_bytes,
            "activation_bytes": activation_bytes,
            "state_bytes": state_bytes_total,
            "adc_samples": adc_samples,
            "dac_samples": dac_samples,
            "n_adc_units": n_adc_units,
            "n_dac_units": n_dac_units,
            "analog_side_params": analog_side_params,
            "digital_side_params": digital_side_params,
            "_source": "jaxpr_verified",
        }

    return {
        "analog_MACs": analog_MACs_total,
        "digital_MACs": digital_MACs_total,
        "sram_bytes": sram_bytes_total,
        "weight_bytes": weight_bytes,
        "activation_bytes": activation_bytes,
        "state_bytes": state_bytes_total,
        "adc_samples": adc_samples,
        "dac_samples": dac_samples,
        "n_adc_units": n_adc_units,
        "n_dac_units": n_dac_units,
        "analog_side_params": analog_side_params,
        "digital_side_params": digital_side_params,
        "_source": "hand_derived_estimate",
    }


# =========== Digital-only PPAC (Accelergy methodology) ===========
def ppac_digital(cfg: Config, adc_bits: int = 8):
    """PPAC for pure-digital INT8 chip at 22nm using Accelergy
    energy-per-action model:
      E = SUM over components (N_actions * E_per_action)

    Per MAC accounting (Sze 2020 breakdown):
      - 1 weight fetch from L1: E_SRAM_READ_8b
      - 1 activation fetch from L1: E_SRAM_READ_8b (unless broadcast-shared)
      - 1 compute MAC: E_INT8_MAC_COMPUTE
      - 1 register accumulate: E_REG_ACCUM
    Plus per-output writeback:
      - 1 SRAM write per output element: E_SRAM_WRITE_8b
    """
    ops = count_ops(cfg)
    total_MACs = ops["analog_MACs"] + ops["digital_MACs"]

    # Per-MAC energy breakdown (Sze 2020):
    E_compute_pJ = total_MACs * E_INT8_MAC_COMPUTE
    E_weight_reads_pJ = total_MACs * E_SRAM_READ_8b
    E_act_reads_pJ = total_MACs * E_SRAM_READ_8b
    E_reg_pJ = total_MACs * E_REG_ACCUM
    # Output writes: activation buffer writes
    E_out_writes_pJ = ops["activation_bytes"] * E_SRAM_WRITE_8b
    # State trajectory writes+reads: within-layer scan intermediate storage
    # For parallel associative_scan, state trajectory is written once and
    # read a few times.  Approximate as 2 reads + 1 write per byte.
    E_state_pJ = ops["state_bytes"] * (2 * E_SRAM_READ_8b + E_SRAM_WRITE_8b)

    E_total_pJ = (
        E_compute_pJ + E_weight_reads_pJ + E_act_reads_pJ
        + E_reg_pJ + E_out_writes_pJ + E_state_pJ
    )

    E_breakdown = {
        "compute_nJ": E_compute_pJ / 1000,
        "weight_read_nJ": E_weight_reads_pJ / 1000,
        "activation_read_nJ": E_act_reads_pJ / 1000,
        "register_nJ": E_reg_pJ / 1000,
        "output_write_nJ": E_out_writes_pJ / 1000,
        "state_trajectory_nJ": E_state_pJ / 1000,
    }

    # Timeloop-simplified latency
    latency_us = total_MACs / (MAC_ARRAY_WIDTH * CLOCK_GHZ * 1000)

    # Area
    A_macarray_um2 = MAC_ARRAY_WIDTH * A_INT8_MAC
    A_sram_um2 = ops["sram_bytes"] * 8 * A_SRAM_BIT
    A_periph_um2 = 0.2 * (A_macarray_um2 + A_sram_um2)
    area_mm2 = (A_macarray_um2 + A_sram_um2 + A_periph_um2) / 1e6

    # Leakage power (Horowitz)
    leakage_mW = area_mm2 * LEAKAGE_mW_per_mm2
    # Dynamic power = E / latency
    P_dyn_mW = (E_total_pJ / (latency_us * 1e-6)) / 1e9
    power_mW = P_dyn_mW + leakage_mW

    cost_usd = WAFER_COST_USD / (DIES_PER_WAFER * YIELD)
    throughput = 1e6 / latency_us

    return {
        "impl": "digital",
        "adc_bits": adc_bits,
        "latency_us": latency_us,
        "throughput_inf_per_s": throughput,
        "energy_per_inf_nJ": E_total_pJ / 1000,
        "power_mW": power_mW,
        "area_mm2": area_mm2,
        "cost_usd_per_chip": cost_usd,
        "sram_KB": ops["sram_bytes"] / 1024,
        "energy_breakdown": E_breakdown,
    }


# =========== Mixed-signal PPAC (NeuroSim methodology) ===========
def ppac_mixed_signal(cfg: Config, adc_bits: int = 8):
    """PPAC for analog crossbar + INT8 digital periphery using
    NeuroSim methodology:
      E_analog   = N_analog_MACs * E_analog_MAC_per_op
      E_ADC      = N_samples * (FoM_ADC * 2^bits)
      E_DAC      = N_samples * (FoM_DAC * 2^bits)
      E_digital  = digital MACs + SRAM (Accelergy)
    """
    ops = count_ops(cfg)

    # NeuroSim energy accounting -- ANALOG side (weight-stationary crossbar):
    # Weights sit on crossbar cells physically, NO per-MAC weight fetch cost.
    # Only compute (analog current summing) is charged per MAC.
    E_analog_pJ = ops["analog_MACs"] * E_ANALOG_MAC

    # ADC energy per sample from Murmann FoM
    E_per_ADC_sample_pJ = FoM_ADC * (2 ** adc_bits)
    E_per_DAC_sample_pJ = FoM_DAC * (2 ** adc_bits)
    E_ADC_pJ = ops["adc_samples"] * E_per_ADC_sample_pJ
    E_DAC_pJ = ops["dac_samples"] * E_per_DAC_sample_pJ

    # DIGITAL side (gate/BN/act) uses full Sze 2020 per-MAC breakdown:
    E_digital_compute_pJ = ops["digital_MACs"] * E_INT8_MAC_COMPUTE
    E_digital_weight_reads_pJ = ops["digital_MACs"] * E_SRAM_READ_8b
    E_digital_act_reads_pJ = ops["digital_MACs"] * E_SRAM_READ_8b
    E_digital_reg_pJ = ops["digital_MACs"] * E_REG_ACCUM

    # Digital-side SRAM: gate weights + activations at ADC boundaries
    sram_bytes_dig = ops["digital_side_params"] + 2 * H * L_SEQ * N_LAYERS
    E_SRAM_writeback_pJ = sram_bytes_dig * E_SRAM_WRITE_8b

    E_total_pJ = (
        E_analog_pJ + E_ADC_pJ + E_DAC_pJ
        + E_digital_compute_pJ + E_digital_weight_reads_pJ
        + E_digital_act_reads_pJ + E_digital_reg_pJ
        + E_SRAM_writeback_pJ
    )

    E_breakdown = {
        "analog_compute_nJ": E_analog_pJ / 1000,
        "adc_nJ": E_ADC_pJ / 1000,
        "dac_nJ": E_DAC_pJ / 1000,
        "digital_compute_nJ": E_digital_compute_pJ / 1000,
        "digital_memory_nJ": (E_digital_weight_reads_pJ +
                              E_digital_act_reads_pJ +
                              E_digital_reg_pJ +
                              E_SRAM_writeback_pJ) / 1000,
    }

    # NeuroSim latency: analog + digital pipeline; ADC settle time
    # per-layer analog time approximated by analog MACs / (parallel width)
    t_analog_per_layer_us = (ops["analog_MACs"] / N_LAYERS) * E_ANALOG_MAC / 1e6
    # ADC: assume 30 ns per sample (typical SAR at 22nm), H samples per crossing
    t_adc_per_layer_us = H * 30e-3  # 30 ns * H samples = 0.03 * H us
    t_digital_per_layer_us = (
        (ops["digital_MACs"] / N_LAYERS) / MAC_ARRAY_WIDTH / (CLOCK_GHZ * 1000)
    )
    t_per_layer_us = max(t_analog_per_layer_us, t_adc_per_layer_us,
                         t_digital_per_layer_us)
    latency_us = N_LAYERS * t_per_layer_us

    # Area (NeuroSim)
    A_crossbar = ops["analog_side_params"] * A_ANALOG_CELL
    A_adc = ops["n_adc_units"] * A_ADC_UNIT * (2 ** adc_bits)
    A_dac = ops["n_dac_units"] * A_DAC_UNIT * (2 ** adc_bits)
    A_dig_mac = 256 * A_INT8_MAC          # smaller MAC array for periphery
    A_sram = sram_bytes_dig * 8 * A_SRAM_BIT
    A_periph = 0.3 * (A_crossbar + A_adc + A_dac + A_dig_mac + A_sram)
    area_mm2 = (A_crossbar + A_adc + A_dac + A_dig_mac + A_sram + A_periph) / 1e6

    leakage_mW = area_mm2 * LEAKAGE_mW_per_mm2
    P_dyn_mW = (E_total_pJ / (latency_us * 1e-6)) / 1e9
    power_mW = P_dyn_mW + leakage_mW

    cost_usd = WAFER_COST_USD / (DIES_PER_WAFER * YIELD) * 1.5  # +50% NRE
    throughput = 1e6 / latency_us

    return {
        "impl": "mixed_signal",
        "adc_bits": adc_bits,
        "latency_us": latency_us,
        "throughput_inf_per_s": throughput,
        "energy_per_inf_nJ": E_total_pJ / 1000,
        "power_mW": power_mW,
        "area_mm2": area_mm2,
        "cost_usd_per_chip": cost_usd,
        "sram_KB": sram_bytes_dig / 1024,
        "energy_breakdown": E_breakdown,
    }


# =========== Report ===========
def fmt(p):
    lines = [
        f"  Latency: {p['latency_us']:8.1f} us | Throughput: {p['throughput_inf_per_s']:7.0f} inf/s",
        f"  Energy:  {p['energy_per_inf_nJ']:8.1f} nJ | Power:      {p['power_mW']:7.2f} mW",
        f"  Area:    {p['area_mm2']:8.3f} mm2| Cost/chip:  ${p['cost_usd_per_chip']:6.2f}",
    ]
    if "energy_breakdown" in p:
        lines.append(f"  Energy breakdown (nJ):")
        for k, v in p["energy_breakdown"].items():
            lines.append(f"    {k:<28s}: {v:8.2f}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adc_bits", type=int, default=8)
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--json", type=str, default="")
    args = p.parse_args()

    print(f"\n=== PPAC (22nm, published methods: Accelergy digital + NeuroSim mixed-signal) ===")
    print(f"    ADC bits = {args.adc_bits}")
    print(f"    FLOP source: JAXPR-verified (walker with 3-way dtype cost)\n")

    all_rows = []
    for cfg in CONFIGS:
        ops = count_ops(cfg)
        source = ops.get("_source", "?")
        source_tag = "[JAXPR]" if source == "jaxpr_verified" else "[estimate]"
        print(f"\n---- {cfg.name} ({cfg.arch}, {cfg.activation}, "
              f"{cfg.params/1000:.1f}K params, test_acc={cfg.train_acc:.4f}) {source_tag} ----")
        d = ppac_digital(cfg, args.adc_bits)
        m = ppac_mixed_signal(cfg, args.adc_bits)
        print(f"\nDIGITAL (Accelergy):")
        print(fmt(d))
        print(f"\nMIXED-SIGNAL (NeuroSim):")
        print(fmt(m))
        print(f"\nMixed-signal vs Digital ratios (<1 = mixed-signal wins):")
        for k in ("latency_us", "energy_per_inf_nJ", "power_mW", "area_mm2"):
            ratio = m[k] / d[k]
            tag = "WIN " if ratio < 1 else "LOSE"
            print(f"  {k:<25s}: {ratio:6.2f}x  [{tag}]")
        all_rows.append({"config": cfg.name, "acc": cfg.train_acc,
                         "digital": d, "mixed_signal": m})

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nWrote {args.json}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "config", "test_acc", "implementation", "adc_bits",
                "latency_us", "throughput_inf_per_s",
                "energy_per_inf_nJ", "power_mW",
                "area_mm2", "cost_usd_per_chip", "sram_KB",
            ])
            for r in all_rows:
                for impl in ("digital", "mixed_signal"):
                    row = r[impl]
                    w.writerow([
                        r["config"], r["acc"], impl, row["adc_bits"],
                        f"{row['latency_us']:.2f}",
                        f"{row['throughput_inf_per_s']:.0f}",
                        f"{row['energy_per_inf_nJ']:.3f}",
                        f"{row['power_mW']:.3f}",
                        f"{row['area_mm2']:.4f}",
                        f"{row['cost_usd_per_chip']:.2f}",
                        f"{row['sram_KB']:.1f}",
                    ])
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
