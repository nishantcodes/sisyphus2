# Embedding-MLP Experiment: High Error Correlation Indicates Feature-Space Alignment

## Result
Torch embedding-MLP with learned day/period embeddings + volume counts achieved CV 0.94819 (±0.00285), **underperforming the champion LGBM (exp 5: 0.95164)** and achieving error correlation of **0.9629** with it — still highly correlated despite the architectural change.

## Why It Failed
The hypothesis was that a smooth-function learner would decorrelate from tree-based splits, especially on the error concentration zone: duration × macro-period interactions (lift 3.9, n≈200–450). However, the 96.3% error correlation reveals the MLP made mistakes on nearly identical rows.

**Root cause:** Learned embeddings for calendar-day features (309 levels) and continuous log-duration don't provide orthogonal information to trees that already split on duration and effectively proxy the temporal patterns via other features. The model solved the same problem the same way — embeddings alone cannot break the learned feature representation.

## Transferable Insight
Error correlation >0.96 signals models are capturing identical signals. Architectural novelty (embedding layers, continuous input handling) doesn't guarantee decorrelation; it requires genuinely different feature engineering or loss objectives. The duration × macro-period cells need a feature class trees haven't seen (e.g., domain-driven interactions, temporal lags, or a loss function targeting calibration in those slices), not just a different learner.
