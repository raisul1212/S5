# Mambino paper — Fig 3 Origin project package

Data files + LabTalk build script for reconstructing Figure 3 (accuracy Pareto)
in OriginPro / OriginLab. Origin's project files (`.opj`, `.opju`) are proprietary
binary — this package uses tab-delimited `.dat` files that Origin's Import Wizard
reads natively, plus a `.ogs` script that automates the workbook + plot build.

## Files

| File | Type | Purpose |
|---|---|---|
| `fig3_panelA_106K_aggregate.dat` | TSV, 2 data rows | Config 4 vs Config 5: mean, SEM, std, n. Used for bar heights + error bars. |
| `fig3_panelA_106K_seeds.dat` | TSV, 16 data rows | Panel A per-seed accuracies (8 seeds × 2 configs). For scatter overlay. |
| `fig3_panelB_188K_aggregate.dat` | TSV, 3 data rows | Corner 1 / Corner 2 / Corner 3′: mean, SEM, std, n. |
| `fig3_panelB_188K_seeds.dat` | TSV, 24 data rows | Panel B per-seed accuracies (8 seeds × 3 configs). |
| `fig3_significance.dat` | TSV, 4 data rows | Paired-t results for the 4 pairwise comparisons. Used to hand-place brackets. |
| `build_fig3.ogs` | LabTalk script | Origin build script — imports .dat files, creates 3 workbooks + 2 base graphs. |
| `README.md` | this file | These instructions. |

Every number traces back to
[`../figure_data/`](../figure_data/) CSVs (git-committed at tag `mambino-paper-v1`)
and to `../../accelergy/LOCKED_master_ppac_document.md` §5a, §5b, §5d.

## Automated build (LabTalk script)

1. Open OriginPro (tested on 2023+; older versions with LabTalk should work).
2. **Edit** `build_fig3.ogs` line 21:
   ```
   string path$ = "C:\Users\raisul\dev\ssm-baselines\S5\paper\origin_project\";
   ```
   Change to wherever this folder actually lives on the machine running Origin.
   Trailing backslash matters. If you copied this folder elsewhere, update accordingly.
3. In Origin: **Window → Script Window** (or press `Alt+3`).
4. Type: `run.file("<full path to build_fig3.ogs>");` and press Enter.
5. Origin builds:
   - `Fig3_PanelA_106K` workbook with `aggregate` + `seeds` sheets → one graph (column plot + scatter overlay)
   - `Fig3_PanelB_188K` workbook with `aggregate` + `seeds` sheets → one graph
   - `Fig3_Significance` workbook with the paired-t table for annotation reference

## Manual build (if the script doesn't work in your Origin version)

For each panel:

1. **File → Import → Import Wizard**.
2. Point to `fig3_panelA_106K_aggregate.dat`.
3. In the Header Lines section, set:
   - **Long Names:** row 1
   - **Units:** row 2
   - **Comments:** row 3
   - **First Data Row:** 4
4. Delimiter: **Tab** (`\t`).
5. Import → creates the workbook.
6. Right-click each column → **Set As** → assign role:
   - `config_label` → **X** (also right-click → **Set Column Values** → Categorical → check "As Category")
   - `mean_peakval` → **Y**
   - `sem_peakval` → **Y Error**
   - `std_peakval`, `n_seeds` → **Disregard**
7. Highlight the Y column → **Plot → Column/Bar/Pie → Column** (or Bar). Origin uses the Y Error column automatically.
8. Import `fig3_panelA_106K_seeds.dat` as a second sheet in the same workbook (repeat step 3–6 with:
   `config_label` → X, `seed_id` → Label, `test_peakval` → Y).
9. Highlight the seeds Y column → **Plot → Scatter** → drag the resulting graph onto the column graph to overlay (or use **Graph → Merge Graph Windows**).

Repeat for Panel B with the 188K files.

## Manual finishing touches (after script or manual build)

The script produces base plots; publication figures need these final tweaks:

**Axes:**
- Y-axis range: **0.575 to 0.625** (matches the paper's clipped range).
- Y-axis tick spacing: **0.01** (major), no minor.
- Y-axis label: `test accuracy @ peak val`.
- Add a break marker at the bottom of the Y-axis (Origin: right-click axis → **Break** → **Show** → position below 0.575).

**Bars:**
- Recommended colors (match Figs 1 and 2 of the paper):
  - Mambino (Config 4, Corner 3′): fill `#d4e6ec`, border `#1c7488`
  - Pure S5 (Config 5, Corner 1): fill `#ece5d4`, border `#8a7a5c`
  - Pure S5 P=16 (Corner 2): fill `#f0d9cf`, border `#a24728`
- Border weight: 1.5 pt.
- Error bar cap width: about 10% of bar width; cap length: 4-6 pt.

**Scatter dots:**
- Symbol: filled circle, size 3-4 pt, edge = surface color for contrast.
- Fill opacity: 50-70% so overlapping dots stay legible.
- Horizontal jitter: about 40-55% of bar width (Origin: **Set Column Values** on X to shift by `col(offset)`).

**Significance brackets** (from `fig3_significance.dat`):

| Panel | Pair | Marker | p-value | Bracket level (1 = lowest) |
|---|---|:---:|---:|:---:|
| A | Config 4 vs Config 5 | `n.s. / *` | 0.064 / 0.032 | 1 |
| B | Corner 1 vs Corner 2 | `**` | 0.005 | 1 |
| B | Corner 3′ vs Corner 2 | `***` | 6.5×10⁻⁵ | 2 |
| B | Corner 1 vs Corner 3′ | `*` | 0.022 | 3 (spans all three bars) |

Add these using Origin's **Text Tool** and **Line/Arrow Tool**:
- Draw a horizontal line between the two bar centers, with short vertical ticks
  down at both ends.
- Above the horizontal line, place the marker (`*`, `**`, `***`) — Symbol font, ~14 pt.
- Below the marker, place the p-value in monospace or a math font.
- Bracket "levels" stack the higher-level brackets further above the bars so they
  don't overlap. Rule of thumb: 10-14 pt vertical spacing per level.

**Legend:** put the family swatches (Mambino / Pure S5 / Pure S5 P=16) in a small
legend box in the upper-left corner of each panel.

**Panel titles:**
- Panel A: **A. 106K-parameter band**
- Panel B: **B. 188K-parameter band**

## Export

Once each panel graph looks right:

- **File → Export Graphs** → Format: **EPS** (or **PDF**) for vector; **PNG** at 300 DPI
  for raster.
- Set page size to match your paper's column width (single column ≈ 85 mm,
  two-column ≈ 180 mm).
- Embed fonts (Origin: Export options → **Embed fonts** checkbox).

## Reference: what the paper's chart should look like

See the HTML reference draft at
[`../fig3_draft.html`](../fig3_draft.html) — it renders the same data with the
target visual style and can be used as a side-by-side while you polish the Origin
version.
