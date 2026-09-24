# Token-Encoded Tabular Features + XGBoost/TinyTokenTransformer Blend for EV Purchase

**Score**: no actual CV or LB score appears in this excerpt — only a constant-prediction baseline (predicting the global positive rate) is printed for AUC/logloss/Brier. Validation is 5-fold, per-class-stratified CV (custom `make_stratified_folds`, seed 2026), scored with a hand-rolled AUC plus logloss/Brier. Any AUC claim from this notebook is unverified until a training-cell output is actually seen.

**Feature engineering** (target-free, fit on pooled train+test): robust numeric scaling (median center, 1st/99th-percentile clip, scaled to [-1,1]); quantile-bin tokens (20 bins) with bin frequency; rounded-value frequency tokens at 2 and 3 decimals (fingerprints exact/near-duplicate numeric values — a hint that s6e9 numerics may be rounded/low-cardinality); categorical value/frequency tokens plus one-hot blocks (cap 64 categories); numeric×numeric and numeric×category interaction tokens; PCA (10 components, fit on an 80k-row pooled sample) for row geometry; leakage-safe target-mean encoding computed **inside each CV fold** with heavy smoothing (500 for single columns, 1500 for pairs). Exact EV-specific column names aren't in the excerpt — `numeric_cols`/`categorical_cols` are inferred dynamically, not hardcoded.

**Models**: XGBoost (`n_estimators=20000`, `learning_rate=0.01`, early-stop 200 — deliberately slow/long) plus a custom TinyTokenTransformer (d_model=96, 2 layers, 4 heads, dropout 0.1, batch 4096, lr 1e-3) attending over the token sequence; final prediction is an OOF-searched blend of the two (grid step 0.01).

**Most worth testing**: the per-fold target-mean tokens (leakage-safe, heavily smoothed) combined with the rounded-decimal frequency tokens — cheap, targeted additions a stock GBDT baseline skips, and plausibly useful if s6e9's numeric columns carry rounding or duplication artifacts.

_source: public notebook `ev-adoption-tokens-xgboost-tinytokentransformer.ipynb`_
