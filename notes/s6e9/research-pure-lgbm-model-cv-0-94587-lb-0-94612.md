# Reverse-Engineered Synthetic-Generator Breakpoints + Dual Target Encoding on LightGBM

**Score**: CV (5-fold OOF) 0.94587, LB 0.94612 — single LightGBM model, no ensembling. Validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, OOF `roc_auc_score`.

**Data facts**: uses an original dataset (`EV_Adoption_and_Range_Anxiety_Dataset.csv`, itzzomkar/ev-adoption-behavior-and-range-anxiety) to compute true per-category/per-value target means. Forum-sourced discovery: `Annual_Income_USD` has synthetic-generator artifacts — a deterministic 100%-positive cliff at `>=170537.0`, a mode-collapse spike at exactly `30000.0`, and a 0%-positive dead zone in `[38000, 42000]`; `Environmental_Concern_Level==1` is an extreme bucket. Drops `Number_of_Cars_Owned` outright and later drops any perfectly-correlated (corr==1.0) or constant columns.

**Feature engineering (exact columns)**: digit extraction per numeric column for exponents k=-4..3 → `{col}_digit{k}` (int8, `(x // 10**k) % 10`); `{col}_org_mean` (mapped from `orig` groupby-target-mean, filled with orig global mean) for any column name shared with the original dataset; numeric→string cast `{col}_cat`; global frequency encoding `{col}_fe` over all categorical + numeric-as-string columns; hand flags `is_30k_spike`, `is_millionaire_cliff`, `is_dead_zone`, `is_env_hater`; per-fold sklearn `TargetEncoder` run twice (smooth='auto' and smooth=10.0) → `{col}_TE_auto`/`{col}_TE_10`, originals dropped.

**Model**: `LGBMClassifier`, n_estimators=40000, lr=0.005, max_depth=5, num_leaves=32, min_child_samples=10, colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0, max_bin=1024, early_stop 700 rounds.

**Most worth testing**: the exact deterministic threshold flags (170537 cliff, 30000 spike, 38–42k dead zone) reverse-engineered from the synthetic generator — a stock GBDT approximates these with splits but can't hit the exact breakpoint a human found by inspection.

_source: public notebook `pure-lgbm-model-cv-0-94587-lb-0-94612.ipynb`_
