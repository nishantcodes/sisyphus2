# Single XGBoost with Measured Digit + Frequency Encoding Ablation

**Score**: OOF AUC 0.94466 (10-fold). Notebook also reports an earlier 5-fold baseline: 0.94184 OOF / 0.94169 LB (before digit/frequency features were added) — a real, if small, CV↔LB gap check. Validation: `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`, but each fold's XGBoost uses a different `random_state`/`seed` = `42+fold` (43–52), a deliberate per-fold seed-diversity trick.

**Ablation (the notable part)**: baseline 0.94204 → +104-digit features claimed → 0.94347 (+0.00143) → +frequency encoding → 0.94466 (+0.00119). Note: the code's `digit_num_cols` list has only 7 numeric columns (56 digit features = 7×8), not the "13 columns / 104 features" the markdown claims — the prose and code disagree, so take the digit count/gain framing with caution even though the AUC deltas are printed.

**Feature engineering (exact)**: `LE_{col}` for Gender/City_Type/Current_Car_Type/Home_Charging_Possible/Subsidy_Available/Range_Anxiety_Level; interactions `Env_Concern_x_Subsidy/Income/Commute`, `Subsidy_x_Income`, `Home_Charging_x_Subsidy`, `Range_Anxiety_x_Subsidy`, `Income_per_Concern`, `Commute_per_Concern`; digit slices `{col}_digit{k}` (k=-4..3) on Age/Annual_Income_USD/Daily_Commute_km/Number_of_Cars_Owned/Charging_Stations_Near_Home/_Near_Work/Environmental_Concern_Level; frequency encoding `{col}_freq` applied to **every** existing column (raw, LE, interaction, and digit) over train+test.

**Model**: single XGBoost (`max_depth=7`, `lr=0.03`, `min_child_weight=5`, `subsample=0.7`, `colsample_bytree=0.5`, `n_estimators=2000`, early-stop 100). No blending — OOF/test arrays saved for later stacking.

**Data facts**: original dataset (`itzzomkar/ev-adoption-behavior-and-range-anxiety`) is loaded and profiled in EDA but never merged or encoded into the model — unused despite being present.

**Most worth testing**: frequency-encoding the *derived* (digit/LE) features, not just raw ones — the notebook's own ablation shows this adds +0.00119 AUC on top of digit features, a clean isolated gain a stock GBDT baseline wouldn't have.

_source: public notebook `s6e9-single-xgb-oof-auc-0-94466.ipynb`_
