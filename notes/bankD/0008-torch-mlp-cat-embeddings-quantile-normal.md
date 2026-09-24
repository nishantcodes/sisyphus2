# Torch MLP with Embeddings: Error Correlation Limits Blending

## Change
Replaced GBDT baseline with a PyTorch MLP using categorical embeddings and quantile-normalized numerics across 3 seeds, targeting the high-loss regions (duration × macro cells, euribor3m, cons.conf.idx).

## Result
**CV regressed to 0.94247 ± 0.00214** vs. champion 0.95047. Blending with champion (50/50) yielded only 0.94933—essentially the champion score. Experiment failed.

## Why It Didn't Work

The critical failure metric is **error correlation (0.9487), not prediction correlation (0.947)**.

- Both models fail on identical hard cases: high-loss decile 9 miscalibration (+0.042), duration bins 200–600s, low cons.conf.idx
- Smooth function approximators and axis-aligned trees make *the same mistakes* on these continuous regions, despite different model families
- The torch MLP's embedding/normalization strategy didn't capture missing interactions (duration × nr.employed, duration × euribor3m lifts of 3.9–4.2)

**Transferable lesson**: When error correlation exceeds 0.94, blending buys <1% lift. The problem isn't feature engineering or architecture—it's that both classes of models hit the same structural wall. Beating this champion requires either (1) an entirely different feature view or (2) a model that handles categorical-continuous interactions fundamentally differently (e.g., attention, deeper embeddings, or domain-specific feature crosses).
