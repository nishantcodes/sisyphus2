# 6-Stream (LGBM/XGB/CatBoost ×2 seeds) Rank Blend with Generator-Artifact Features

**Score**: no explicit "CV: X / LB: Y" line is printed in the text shown — scores are computed at runtime via `roc_auc_score` per stream and for the final blend, and the filename implies ~0.9459, but that number isn't confirmed by an actual printed value here. Validation: `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`, OOF AUC per stream and for the final blend.

**Data facts**: appends ~10,000 real rows from `itzzomkar/ev-adoption-behavior-and-range-anxiety` directly into the training pool (row concat on common columns, not just mean-mapping) for a claimed 678,665-row training set. Exploits known synthetic-generator (CTGAN) artifacts: `Annual_Income_USD` floor at exactly 30000, `Daily_Commute_km` floor at 5.0 — both framed as generator boundary cutoffs, consistent with other notebooks' "mode collapse spike" finding.

**Feature engineering (exact)**: `LE_{col}` label-encodes Gender/City_Type/Current_Car_Type/Home_Charging_Possible/Subsidy_Available/Range_Anxiety_Level; floor/trailing-zero flags `inc_at_floor`, `inc_round_100/1000`, `inc_tz`, `inc_last_digit`, `inc_mod_100/1000`, `km_at_floor`, `km_round_1`, `km_last_digit`, `km_mod_100`; multi-resolution quantization `inc_div_{100,250,500,1000,2000,5000}`, `km_div_{5,10,25,50}`; domain interactions (`Env_Concern_x_Subsidy/Income`, `Subsidy_x_Income`, `Affordability_Per_Car`, `Income_Per_Commute`, `Charging_Density_Per_Commute`, etc.); 5 joint integer keys; 56 digit-slice features `{col}_digit{k}` (k=-4..3) over the 7 numeric fields; transductive frequency `{col}_freq`; cross-fitted (per-fold) Bayesian target encoding (m=100/50) on 4 numeric-divisor keys plus 5 joint keys.

**Models**: LightGBM (depth=6, leaves=24, lr=.015, colsample=.35), XGBoost (depth=5, lr=.015, colsample=.70, min_child_weight=15), CatBoost (depth=5, lr=.035, l2=3.0) — each ×2 seeds (42/202), 10-fold.

**Most worth testing**: the Nelder-Mead-optimized rank-space blend across 6 correlated-but-distinct streams, layered on generator-artifact features a stock single-GBDT baseline has neither of.

_source: public notebook `s6e9-94-59-transformer-and-gbdt-ensemble.ipynb`_
