# Fold-Safe Target Encoding of Grid Values + Concern Crosses

**Change:** Fold-safe (out-of-fold) target encoding applied to the two coarse-grid synthetic numerics (Annual_Income_USD: 13,214 unique values, Daily_Commute_km: 805 unique) plus explicit crosses of Environmental_Concern_Level with Subsidy_Available and with income bands.

**Result:** CV 0.94530 ± 0.00081 — new best, clearing the prior 0.9437 (engineered-ratio XGB) and well outside the ±0.0009 fold-noise band that capped three raw GBDT baselines.

**Why it worked (mechanism, not score):** Error analysis on the MLP champion showed a specific, sharp failure mode: Environmental_Concern_Level=4/5 crossed with Subsidy_Available carried loss lift 2.4–2.8 on ~175k rows, and crossed with narrow income bands carried lift ~2.0–2.2 on 12-14k-row cells. These are per-cell base-rate effects on a *coarse generator grid* (only 13k distinct income values across 668k rows) — exactly the shape trees approximate poorly via splits (each split only halves a region, never directly encodes "this exact grid cell's average outcome"). Target encoding hands the model the cell's outcome rate directly, and doing it fold-safe avoids the leakage that would have inflated CV without real signal. The jump beyond fold noise confirms the grid values encode real generator structure, not just noise.

**Transferable takeaway:** When error analysis finds high-lift *narrow categorical/numeric-bucket interactions* rather than smooth misses, target/frequency encoding of those exact cells beats manual ratio features or a different architecture. Next: check for interaction encodings on the still-unaddressed Subsidy×Income and Commute-grid cells, and re-test whether GBM plateau/NN-decorrelation logic still holds with these new features.
