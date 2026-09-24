# Statistical Significance Testing for CV Score Gains (Content Not Shown)

**Score**: no CV or LB score appears anywhere in this excerpt — the notebook opens with only a markdown title/author block, a static image, and a library-import cell (numpy, pandas, xgboost is *not* imported; sklearn's `StratifiedKFold`/model-selection, `matplotlib`, `seaborn` are). No data loading, feature engineering, model training, or metric-printing code is included in what was provided.

**What can be inferred from the title alone**: this notebook's stated purpose ("how to tell a real gain from noise") is a methodology piece about distinguishing genuine CV improvements from CV-fold noise — likely via bootstrap/permutation-style significance testing of AUC deltas, a useful discipline given how many s6e9 notebooks report tiny (0.001–0.002) AUC gains from individual features without any noise check. But this is inference from the title, not something the shown code demonstrates.

**Validation scheme, feature engineering, models, and data facts**: none are visible in the supplied content — nothing beyond the import cell has rendered. Do not treat any score, feature, or technique as confirmed for this notebook based on the current excerpt.

**Most worth testing**: unknown from this excerpt — flagging one specific idea "most worth testing" would be fabricating conclusions the notebook hasn't shown yet. If more cells (the actual significance-testing method) can be supplied, a substantive note can be written; right now there is nothing concrete to extract beyond the title's framing.

_source: public notebook `s6e9-starter-how-to-tell-a-real-gain-from-noise.ipynb`_
