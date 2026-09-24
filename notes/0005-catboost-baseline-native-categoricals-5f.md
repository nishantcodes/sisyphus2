# CatBoost Baseline — Native Categoricals, Ordered Boosting, Shared 5-Fold CV

**Change:** CatBoost with native categorical handling and ordered boosting (its target-leakage-resistant training scheme), same shared 5-fold splits as LGBM and XGBoost baselines.

**Result:** CV 0.94133 ± 0.00077 — the third GBM library, and the third near-identical score (LGBM 0.94185, XGBoost 0.94151, CatBoost 0.94133). All three sit within ~0.0005 of each other, well inside fold-to-fold noise.

**Why this matters (mechanism, not score):** CatBoost's ordered boosting was specifically expected to find *different* interactions than LGBM/XGBoost because it processes categorical statistics without the leakage that standard target encoding introduces — if the other two baselines were overfitting subtly to categorical target leakage, CatBoost should have scored differently (better or worse). It didn't. That's a strong signal the ceiling isn't coming from categorical-encoding leakage or interaction-finding strategy at all — it's the raw feature set itself that caps performance around 0.9415-0.942.

**Why the ceiling likely holds:** Three structurally different boosting algorithms (leaf-wise histogram, level-wise exact/approx, ordered/symmetric-tree) converging this tightly closes off "try another GBM" as a productive direction entirely.

**Transferable takeaway:** Stop iterating on GBM library choice. Next turn should pursue feature engineering on the flagged segments (income bands, charging availability, city type) or ensembling the three OOF arrays, not a fourth tree-based baseline.
