# Nested per-cell kNN target stats → raw-feature LGBM (exp67): No improvement, diagnostic clear

**Result:** CV 0.94175 ± 0.00093 vs. champion 0.94597 ± 0.00078. Lost ~42 points.

**What changed:** Replaced axis-aligned TE/digit feature engineering with nested cKDTree kNN target statistics (per-cell multi-numeric density estimates), then LGBM on raw features + those stats. The hypothesis: remaining loss in Environmental_Concern × {Subsidy, Income bands} cells (lift 2.45–2.80) is fine joint structure in numeric space that axis-aligned splits miss.

**Why it failed:** The experiment confirmed the opposite — loss is *not* in numeric joint structure. Evidence:
1. Champion's calibration is already tight (all decile gaps <0.004 except one: +0.0039 in decile 8), so variance-prone kNN stats added noise faster than they captured signal.
2. Error analysis shows loss concentrates in *categorical* interactions (Concern=4/5 × Subsidy=Yes/Income_bands), not numeric fine structure.
3. Disagreement with exp#65 (neural model) is sharp on 1% of rows but in sparse regions; the main loss cells are well-defined categories, not numeric outliers.

**Closed path:** Per-cell local density in numeric space is the wrong probe. Next: explicit categorical interaction terms or cell-level target-rate smoothing within the {Concern=4/5} × {Subsidy, Income} strata.
