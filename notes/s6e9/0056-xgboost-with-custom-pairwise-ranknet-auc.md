# RankNet Loss Did Not Beat Log-Loss; The Bottleneck Was Feature Engineering

**Change:** XGBoost with pairwise RankNet objective optimizing AUC directly, instead of log-loss.

**CV Result:** 0.94382 (baseline, worse than champion 0.94570). Loss function swap failed.

**Why it didn't work:**

The hypothesis was plausible but misdirected. Yes, the champion shows terrible calibration in deciles 1–7 (gaps −0.46 to −0.56), but those regions aren't the AUC bottleneck — actual rates are 0.0001–0.0797, so even wild overprediction there preserves correct ordering. Log-loss isn't wasting capacity on bad ranking in the tail; it's calibrating a correctly-ordered tail.

The real problem, per error analysis: **missing interactions**. Concern=3 × Subsidy=Yes (lift 1.60, 75k rows) and Concern=5 × income bands (lift 1.17–1.32) show the model has no feature or split capturing these signals — no loss function fixes that. The only decorrelated model (exp #42) achieves 0.0012 error correlation by explicitly engineering Concern×Subsidy as a target, not by changing the loss.

**Transferable lesson:** When AUC is good but calibration is poor, the bottleneck is upstream (features, interactions, binning), not the loss function. Pairwise loss is high-variance and doesn't add signal where it doesn't exist. RankNet works if ranking is the problem; here, it was feature engineering.
