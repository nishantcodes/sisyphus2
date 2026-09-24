# LGBM + Fold-Safe Grid/Interaction Target Encoding

**Change:** Ported the fold-safe target encoding (Annual_Income_USD, Daily_Commute_km grid values, Concern×Subsidy crosses) from the XGB winner onto LGBM, plus a new three-way Concern×Subsidy×income-bin cell that had never been encoded before.

**Result:** CV 0.94548 ± 0.00077, up from LGBM's raw-feature baseline of 0.94185 — a ~0.0036 jump, well outside the ±0.0009 fold-noise band, and now the joint best alongside the XGB TE model (0.94530).

**Why it worked (mechanism, not score):** The LGBM error analysis showed the identical failure signature as XGB's: Concern=4/5 × Subsidy=Yes carrying loss lift 2.45–2.80 on 80-95k-row slices, and Concern × narrow income-grid bands carrying lift ~2.0–2.24 on 12-14k-row cells. This confirms the failure is *feature-representation-level*, not learner-specific — both tree libraries independently under-fit the same coarse-grid interaction cells because splits can only bisect regions, not state a cell's outcome rate directly. Encoding it fixes it regardless of which GBM consumes the feature.

**Why this matters for blending:** Despite the CV gain, disagreement analysis shows LGBM and CatBoost are still near-identical (error correlation 0.996) — the grid-TE fix is now the shared baseline improvement across the whole GBDT family, not a source of diversity. Any future blend needs a genuinely different feature view or architecture, not another TE-boosted tree model.

**Transferable takeaway:** A representation fix validated on one learner (XGB) reliably transfers to sibling tree learners when the error analysis confirms it's the same interaction gap — check error analysis correlation before assuming a new model adds blend value.
