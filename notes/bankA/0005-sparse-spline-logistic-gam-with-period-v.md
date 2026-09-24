# Period-Varying Duration GAM: Architecture Mismatch, Not Feature Gap

**Change**: Replaced LGBM with sparse spline logistic GAM; modeled duration curves separately for each macro period (euribor3m, nr.employed, cons.conf.idx as period proxies).

**Result**: CV dropped to 0.94315 ± 0.00265 vs. champion 0.94947 ± 0.00296. Hypothesis failed.

**Why it didn't work**: The error analysis showed high lift (loss concentration) in cells where duration crossed with macro features (e.g., duration ∈ [221, 279] AND cons.conf.idx ∈ [-36.1, -26.9], lift 4.01). This *suggested* a missing interaction. However, the GAM's errors correlated 0.954 with the champion's errors in those same cells—nearly identical mistakes on identical rows.

**Mechanism**: When two vastly different architectures (tree boosting vs. smooth regression) fail on the same rows with the same error pattern, the problem isn't an *interaction* the first model missed. It's either: (1) genuine prediction noise—those cells simply contain less signal; (2) missing *features* entirely (not feature engineering); or (3) the interaction is more complex than any single feature decomposition captures. Fitting separate duration curves for each period didn't help because duration curves weren't the bottleneck.

**Takeaway for future work**: High error concentration + high inter-model error correlation = feature gap or noise floor, not modeling gap. Adding architectural flexibility won't cut loss; add exogenous features instead.
