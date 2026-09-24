# XGBoost + Engineered Ratios (Affordability/Charging/Anxiety) + Count Encodings

**Change:** Added explicit ratio and interaction features to XGB — affordability (income vs. commute/cars), charging access (home charging flag × station counts), range anxiety (commute distance vs. some range/battery proxy) — plus count/frequency encodings for the two near-continuous synthetic columns.

**Result:** CV 0.94370 ± 0.00092, up from the 0.9413–0.9419 GBM plateau — a real jump, clearing the fold-noise band (±~0.0008) that three raw-feature baselines were stuck inside.

**Why it worked (mechanism, not score):** The prior three baselines (LGBM/XGB/CatBoost) converged tightly on raw columns, which proved trees weren't the bottleneck — representation was. Trees split on single features or learn interactions only through many sequential splits, which is a lossy, sample-hungry way to approximate a *ratio*. Explicitly computing affordability, charging-access, and anxiety ratios hands the model pre-formed nonlinear combinations in one column each, so a single split can capture what previously took a deep chain of splits (and more data) to approximate. This aligns with the large LB headroom (0.94631 top vs 0.94166 prior) — top teams were likely exploiting exactly this kind of domain-ratio feature engineering, not a better learner.

**Transferable takeaway:** Feature-space ratios/interactions on domain-meaningful drivers (affordability, access, anxiety) are the productive lever here, not learner choice or hyperparameters. Next turn: push further on interaction depth (e.g., three-way combos) or try these same engineered features on LGBM/CatBoost to see if the gain transfers across learners (confirming it's feature-driven, not XGB-specific).
