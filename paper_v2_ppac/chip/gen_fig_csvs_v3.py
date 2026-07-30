"""Regenerate fig4_v3 + fig5_v3 CSVs from the hybrid v3 pipeline.

Hybrid = SCALE-Sim cycles (cycle-accurate, includes fill/drain) + direct_ppac
energy (per-component from Accelergy ERT).

Adds pipelined-Mambino variants with correct iso-params baselines:
  Config 4  (Mambino gelu 106K, P=8)          -> Config 5 (Pure S5 gelu 106K, P=16)
  Corner 3' (Mambino h_glu2 lr 188K, P=8)     -> Corner 2 (Pure S5 h_glu2 lr 188K, P=16)

Corner 1 (Pure S5 half_glu2 full-rank 188K) is included as the "S5 reproduction
sanity check" line -- no predictor, no pipelining claim, sequential = pipelined.
"""
import csv, json, os
from pathlib import Path

CHIP = Path(__file__).resolve().parent
FIGS = CHIP.parent.parent / "paper" / "figure_data"
FIGS.mkdir(parents=True, exist_ok=True)

LABEL = {'config4':'Config 4','config5':'Config 5','corner1':'Corner 1',
         'corner2':'Corner 2','corner3p':"Corner 3'"}
FAMILY = {'config4':'Mambino','config5':'Pure S5','corner1':'Pure S5',
          'corner2':'Pure S5','corner3p':'Mambino'}
BAND = {'config4':'106K','config5':'106K','corner1':'188K',
        'corner2':'188K','corner3p':'188K'}
ROLE = {'config4':'Mambino gelu 106K',
        'config5':'Pure S5 gelu 106K (iso-params NoGLU baseline)',
        'corner1':'Pure S5 half_glu2 full-rank 188K (published S5 reproduction)',
        'corner2':'Pure S5 half_glu2 low-rank 188K (iso-params, iso-gate-topology baseline)',
        'corner3p':"Mambino half_glu2 low-rank 188K (hero)"}

# Pipelined pairing: iso-params, iso-gate-topology Pure S5 baseline
SAME_P_PAIR = {'config4': 'config5', 'corner3p': 'corner2'}

# --- Load hybrid inputs ---
# Energy from direct_ppac.py v3 (per_config_ppac_v3.json)
v3 = json.load(open(CHIP / "per_config_ppac_v3.json"))
# Cycles from SCALE-Sim sweep (scalesim_sweep_per_config.csv)
ss_rows = list(csv.DictReader(open(CHIP / "scalesim_sweep_per_config.csv")))
ss_cycles = {(r['config'], int(r['array_x']), int(r['array_y'])): int(r['total_cycles'])
             for r in ss_rows}

# --- Build hybrid records ---
FREQ_HZ = 1e9
CONFIGS = ['config4','config5','corner1','corner2','corner3p']
ARRAY_SIZES = [(8,8),(16,16),(32,32),(64,64)]

# Precompute per-cfg per-arr hybrid metrics
hybrid = {}
for cfg in CONFIGS:
    hybrid[cfg] = {}
    for ax, ay in ARRAY_SIZES:
        key = f"{ax}x{ay}"
        d = v3[cfg][key]
        cyc = ss_cycles[(cfg, ax, ay)]
        lat_s = cyc / FREQ_HZ
        energy_mJ = d['energy_mJ']
        power_mW = (energy_mJ / 1000) / lat_s * 1000
        tps = 1 / lat_s
        acc = d['accuracy']
        edp = energy_mJ * (lat_s * 1000)                        # mJ * ms
        ed2p = energy_mJ * (lat_s * 1000) ** 2                  # mJ * ms^2
        hybrid[cfg][key] = dict(
            config=cfg, array_x=ax, array_y=ay, n_pe=ax*ay,
            accuracy=acc, energy_mJ=energy_mJ, latency_ms=lat_s*1000,
            power_mW=power_mW, throughput_ips=tps, total_cycles=cyc,
            pj_per_real_mac=d['pj_per_real_mac'],
            acc_per_mJ=acc / energy_mJ,
            acc_x_throughput_per_W=acc * tps / (power_mW/1000),
            edp_mJ_ms=edp, ed2p_mJ_ms2=ed2p,
            acc_over_edp=acc / edp,
        )

# --- Compute pipelined variants ---
# Config 4 & Corner 3' get pipelining relative to their iso-params Pure S5 baseline;
# Configs 5, Corner 1, Corner 2 have no predictor branch -> pipe == seq (shown for context).
def pipe_cycles(cfg, arr_key, seq_cyc):
    if cfg not in SAME_P_PAIR:
        return seq_cyc                                    # no predictor -> no pipelining
    base_cfg = SAME_P_PAIR[cfg]
    base_cyc = hybrid[base_cfg][arr_key]['total_cycles']
    excess = seq_cyc - base_cyc
    if excess > 0:
        return seq_cyc - int(0.75 * excess)               # 75% overhead recovery
    return seq_cyc                                        # baseline is already slower

fig4_rows = []
for cfg in CONFIGS:
    for ax, ay in ARRAY_SIZES:
        key = f"{ax}x{ay}"
        r = hybrid[cfg][key]
        seq = r['total_cycles']
        pipe = pipe_cycles(cfg, key, seq)
        pipe_lat_ms = pipe / FREQ_HZ * 1000
        energy = r['energy_mJ']
        pipe_pwr_mW = (energy / 1000) / (pipe / FREQ_HZ) * 1000 if pipe > 0 else 0
        pipe_tps = 1e9 / pipe if pipe > 0 else 0
        pipe_edp = energy * pipe_lat_ms
        r_ext = dict(r,
            label=LABEL[cfg], family=FAMILY[cfg], params_band=BAND[cfg], role=ROLE[cfg],
            baseline_for_pipe=SAME_P_PAIR.get(cfg, '-'),
            pipe_cycles=pipe, pipe_latency_ms=pipe_lat_ms,
            pipe_power_mW=pipe_pwr_mW, pipe_throughput_ips=pipe_tps,
            pipe_edp_mJ_ms=pipe_edp,
            pipe_acc_per_edp=r['accuracy'] / pipe_edp if pipe_edp > 0 else 0,
            pipe_acc_x_tps_per_W=r['accuracy'] * pipe_tps / (pipe_pwr_mW / 1000) if pipe_pwr_mW > 0 else 0,
        )
        for k in ('energy_mJ','latency_ms','power_mW','throughput_ips','pj_per_real_mac',
                  'acc_per_mJ','acc_x_throughput_per_W','edp_mJ_ms','ed2p_mJ_ms2',
                  'acc_over_edp','pipe_latency_ms','pipe_power_mW','pipe_throughput_ips',
                  'pipe_edp_mJ_ms','pipe_acc_per_edp','pipe_acc_x_tps_per_W'):
            r_ext[k] = round(r_ext[k], 5)
        fig4_rows.append(r_ext)

# --- fig4 CSV ---
fig4_path = FIGS / "fig4_v3_chip_pareto.csv"
hdr = ['config','label','family','params_band','role','accuracy',
       'array_x','array_y','n_pe',
       'total_cycles','latency_ms','energy_mJ','power_mW','throughput_ips',
       'pj_per_real_mac','acc_per_mJ','acc_x_throughput_per_W',
       'edp_mJ_ms','ed2p_mJ_ms2','acc_over_edp',
       'baseline_for_pipe','pipe_cycles','pipe_latency_ms','pipe_power_mW',
       'pipe_throughput_ips','pipe_edp_mJ_ms','pipe_acc_per_edp','pipe_acc_x_tps_per_W']
with open(fig4_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(hdr)
    for r in fig4_rows:
        w.writerow([r[k] for k in hdr])
print(f"wrote {fig4_path} ({len(fig4_rows)} rows)")

# --- fig5 CSV — per-component energy breakdown from v3 (FIXES the __ARITH__ key bug) ---
fig5_path = FIGS / "fig5_v3_energy_breakdown.csv"
# Component set from v3 (already-correct names): mac, weight_sram, act_sram,
# weights_spad, psum_spad, state_sram, elemwise_{class}, leakage
comp_names = ['mac','weight_sram','act_sram','weights_spad','psum_spad','state_sram',
              'elemwise_trivial','elemwise_moderate','elemwise_transcendental',
              'elemwise_reduction','elemwise_structural','leakage']
with open(fig5_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['config','label','family','params_band','array_x','array_y','n_pe',
                'total_energy_mJ'] + [f"{c}_mJ" for c in comp_names])
    for cfg in CONFIGS:
        for ax, ay in ARRAY_SIZES:
            key = f"{ax}x{ay}"
            d = v3[cfg][key]
            row = [cfg, LABEL[cfg], FAMILY[cfg], BAND[cfg], ax, ay, ax*ay,
                   round(d['energy_mJ'], 5)]
            for c in comp_names:
                pJ = d['per_component_pJ'].get(c, 0.0)
                row.append(round(pJ / 1e9, 6))       # pJ -> mJ
            w.writerow(row)
print(f"wrote {fig5_path}")

# --- Report the pipelined comparison table at 16x16 ---
print(f"\n=== Pipelined @ 16x16 (SCALE-Sim cycles + v3 energy) ===")
print(f"{'Config':<10}{'Role':<50}{'seq_lat':>10}{'pipe_lat':>10}{'seq_edp':>10}{'pipe_edp':>10}"
      f"{'acc/mJ':>10}")
print('-'*118)
for r in fig4_rows:
    if r['array_x'] != 16: continue
    print(f"{r['config']:<10}{r['role'][:50]:<50}"
          f"{r['latency_ms']:>8.3f}ms{r['pipe_latency_ms']:>8.3f}ms"
          f"{r['edp_mJ_ms']:>10.4f}{r['pipe_edp_mJ_ms']:>10.4f}"
          f"{r['acc_per_mJ']:>10.3f}")
