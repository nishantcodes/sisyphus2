# XGBoost Baseline — Native Categoricals, Shared 5-Fold CV

**Change:** XGBoost with native categorical support (`enable_categorical`), same shared 5-fold splits as the LGBM baseline, no target/one-hot encoding.

**Result:** CV 0.94151 ± 0.00076 — essentially tied with LGBM (0.94185 ± 0.00093), within noise of each other.

**Why this matters (mechanism, not score):** Two different boosting implementations, both using native categorical splitting, converge to nearly identical CV. That's evidence the ceiling here is being set by the *information content of the raw features*, not by which tree-boosting library or split algorithm is used — i.e., this isn't a "pick the right GBM" problem. The tighter std (±0.00076 vs ±0.00093) suggests XGBoost's categorical handling is slightly more stable across folds, but the gap is too small to call a real difference.

**Why it likely didn't move the needle:** Both models see the same raw signal; neither does feature engineering on the flagged weak segments (income bands, charging availability, city type). Convergence across families implies future gains won't come from swapping GBM libraries — they'll need engineered interactions/encodings targeting those segments, or a fundamentally different signal source (external data, non-tree model).

**Transferable takeaway:** Don't spend more turns on baseline GBM variants; the two-model agreement closes that line of investigation. Redirect effort to feature engineering on the error-analysis segments or a differently-shaped model (e.g., linear/NN capturing different interactions).
