# Sony Faculty Innovation Award — proposal strategy (Mambino 2.0 + Cluster B)

**Purpose.** Capture the full strategy from the 2026-07-30 session so the proposal can be
picked up cold. This is strategy + honest evidence state, NOT a drafted proposal yet.

---

## The award
- **Sony Research Award Program → Faculty Innovation Award.** Up to **$100K, 1 year**. Deadline **Sept 15, 2026** (~6.5 wks out as of 2026-07-30).
- **Eligibility CLEARED:** Raisul is a full-time **Purdue professor** → valid PI (full/assoc/assistant all qualify; must supervise PhD students — yes). Gilbreth is Purdue's cluster.
- **Open call** — NO pre-existing Sony contact needed; the program pairs the awardee with a Sony R&D group. (This is why Faculty Innovation > Focused Research, which is pre-arranged co-development.)

## Direction (LOCKED): on-sensor / event-based vision edge AI
**Chosen over RF sensing and language/translation.** RF is a Sony *research keyword* but NOT a core product → weak alignment. Vision on-sensor AI = Sony's **#1 semiconductor business**, verified real products:
- **IMX500 "Intelligent Vision Sensor"** — on-chip AI accelerator + SRAM in one stacked chip; runs inference ON the sensor, outputs metadata not images (low-bandwidth). Platform: **AITRIOS**.
- **Event-Based Vision Sensors (IMX636/637, w/ Prophesee)** — output only *changes*, low-latency/low-bandwidth = **surprise implemented in silicon**.
- **2026:** Sony–Mitsubishi Electric JV for AI vision sensors (factory automation/inspection).
- **Keyword:** primary **"Co-design of Sensor and Physical AI"**; alt "Ultra-low-power/Low-bandwidth Sensing" or "Sensor-aware Minimal-footprint Neural Imaging" (all IT → Computer Vision).

## The thesis (what stands out — spend boldness HERE)
- **NOT efficiency** — most crowded lane in Sony's inbox (pruning/quant/NAS/tiny-nets/SNNs). Efficiency = table stakes, cannot be the headline.
- **Hero line:** *"compute ∝ surprise + on-device learning, co-designed with event vision sensors."*
- Three genuine differentiators: **(1) adaptive computation** (energy scales with input surprise, not a fixed compressed model) — a different *category* from the efficiency crowd, and a natural match to event sensors that are already input-adaptive; **(2) on-device learning after deployment** (Cluster B); **(3) a verified, interpretable brain-inspired mechanism** (not a black-box trick).
- The **adaptive-model + adaptive-sensor pairing** is the memorable, hard-to-copy angle.

## Delivery structure (ambition vs deliverability — the load-bearing design)
Layer so the proposal **cannot fully fail**:
| layer | risk | status |
|---|---|---|
| surprise mechanism → surprise-as-detector on vision | LOW | validated on language; **future-frame-prediction anomaly detection is an established method** (field-supported) |
| **surprise-gated adaptive compute (energy-vs-surprise curve)** | MED | buildable, still distinctive — **the de-risked CORE/spine** |
| Cluster-B on-device adaptation | HIGH | the **moonshot, scoped with a fallback** (zero code today) |
If Cluster B underperforms, detector + adaptive-compute is still a complete, publishable, Sony-relevant contribution. Ambition and credibility reinforce because the worst case is still a win.

## HONEST evidence state — DO NOT OVERCLAIM (this is competitive; overclaims get caught)
- **PROVEN:** surprise signal meaningful/calibrated — **2.19 z-unit separation, 8/8 seeds, on LANGUAGE**; gate improves acc/param on ListOps.
- **HYPOTHESIS, untested by us:** (a) surprise-as-detector on vision/events; (b) surprise-gated **adaptive compute** — **NEVER built**: the gate today modulates the *write* (scalar multiply), NOT *compute* (skipping layers) — "energy ∝ surprise" is pure hypothesis; (c) Cluster B — zero code.
- **3 of 4 pillars are unproven in-domain.** The advantage is the *research question*, not a demonstrated result. Frame it that way.

## Competition / risks (must be answered in the proposal)
- **SNNs** = incumbent event-driven adaptive-compute for event cameras. Differentiate: **trainable by backprop, calibrated/interpretable surprise, dense-HW-friendly** vs SNN training difficulty. ("Why not an SNN?" will be asked.)
- **Redundancy risk (the crux):** the event sensor already gates on low-level change — does model-level *semantic* surprise add *measurable* savings on top? Untested; if not, the story collapses.
- **TTA / continual learning** exist — Cluster B needs explicit differentiation.

## THE de-risking action (do BEFORE submitting)
Run a **preliminary event/vision surprise-as-detector test on 2×A30**: small event or video-anomaly set (DVS-Gesture, or UCSD-Ped/Avenue via next-frame prediction), adapt the `eps_diag.py` machinery to a visual stream; show surprise separates events/anomalies on VISUAL data + a first rough energy-vs-surprise measurement. Converts one hypothesis → in-domain preliminary evidence. Signal → grounded, fundable proposal ("we already see it working"); no signal → pivot before staking the proposal.

## Two-tier resource plan (the compute ask is a STRENGTH, not a constraint)
- **Preliminary (2×A30, now):** mechanism validated (language) + small vision demo = credibility.
- **Proposed year-1 (funded, ambitious):** scale to real event-vision (Prophesee Gen1/Gen4, DSEC, larger video-anomaly); full 3 experiments; on-sensor budget characterization (simulate IMX500 envelope; ideally real HW via collaboration).
- **Resource ASK = part of the proposal:** cloud/GPU compute (budget line) + **Sony hardware/platform access (IMX500/EVS dev kits, AITRIOS, event datasets)** = the two-way collaboration hook (Sony provides sensor+platform, PI provides the efficient surprise-gated model).

## Three experiments (modality-agnostic; instantiate on event/vision)
1. **Surprise-as-detector** — surprise separates events/anomalies; AUC/AUROC; baseline = autoencoder reconstruction error (show the trained calibrated surprise beats raw recon error).
2. **Surprise-gated adaptive compute** — HEADLINE **energy-vs-surprise curve** (Mambino energy scales with surprise; fixed SSM/Transformer constant); energy–vs–detection-quality Pareto; guardrail: must not sleep through event *onsets*.
3. **Cluster-B on-device adaptation** — new environment/deployment; adapt "normal" on-device from unlabeled stream (no gradients/labels); vs no-adapt baseline and full-finetune upper bound.

## Year-1 milestones
M1 surprise-detector on vision · M2 gated adaptive compute + energy characterization (headline) · M3 Cluster-B cross-deployment adaptation · M4 on-sensor deployment characterization + Sony demo.

## Next steps
1. **Run the preliminary event/vision test** (de-risks the whole proposal).
2. Draft the **one-pager**: bold single-thesis hero → de-risked core → scoped moonshot+fallback → preliminary evidence → year-1 plan + resource/collaboration ask → budget.
3. Watch the timeline (deadline Sept 15, 2026).

Related: [[cluster_a_surprise_gate]], [[mambino2p0_ppac_result]], [[cluster_b_fastweight_memo]], [[charlm_plan]], [[user_name]] (Purdue PI), [[user_long_term_vision]].
