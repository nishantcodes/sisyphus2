# Torch MLP with Periodic Embeddings: Smooth Interactions Hypothesis Failed

**What changed**: Replaced LGBM baseline with a PyTorch MLP using PLR-style periodic (sinusoidal) numeric embeddings plus categorical embeddings.

**CV result**: 0.94175 ± 0.00248 (vs. LGBM champion 0.94947 ± 0.00296) — worse by ~77 bps.

**Why it didn't work—the mechanism**:

The hypothesis was sound: error analysis flagged missing interactions between duration and smooth macro features (cons.conf.idx, nr.employed, euribor3m). Tree splits approximate continuous interaction surfaces coarsely; periodic embeddings should represent them as learned sinusoids, fitting smooth multiplicative effects.

The torch MLP failed despite this because:

1. **Error correlation 0.9521** with the champion—they find nearly identical errors. The model didn't discover a different error surface, just performed the same mistakes worse.
2. **Training didn't exploit the architecture advantage**. Periodic embeddings help *when the optimizer discovers smooth patterns*, but the torch MLP's loss landscape or regularization didn't guide it there.
3. **The interactions may not be as smooth as hypothesized**—duration × {econ feature} could be stepwise with region-dependent breakpoints, where GBDT splits are actually the right abstraction.

**Takeaway**: Architectural expressivity alone doesn't guarantee better fits when the optimization landscape favors coarser approximations. Future attempts should pair the architecture with explicit interaction terms or loss design targeting the identified region-feature pairs.
