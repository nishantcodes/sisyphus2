# PLR-Embedding MLP on Exp 9 Features: Neural Net Underperforms on Feature-Matched Test

## What Happened
Tested an MLP with periodic (PLR) numeric embeddings using experiment 9's hand-crafted 42-feature set (4 seeds/fold). Result: **CV 0.95096 ± 0.00292**, underperforming the LightGBM champion (0.95135) by ~0.39 basis points.

## Why This Matters
The experiment was designed to isolate whether neural nets are architecturally weak on this problem or simply starved of features. By feeding the MLP *identical* features to the champion, we separated the confound: exp 14 (NN on raw features) lost 0.51bp, but exp 9 features were hand-crafted same-day aggregates (+0.19bp gain). If the MLP matched the champion with the rich feature set, the architecture would be in play; if it still lagged, GBDT has a structural advantage here.

## The Verdict
**GBDT wins on inductive bias.** Even with periodic embeddings—a principled numeric representation—the MLP couldn't close the gap. The error analysis shows all models are highly correlated (0.94+ pred corr, 0.94+ error corr), meaning they're learning the same patterns. The MLP's loss is not decorrelated enough to justify ensemble inclusion. Takeaway: this problem favors tree-based splits and feature interactions over learned embeddings; further neural net gains would require fundamentally different architectures (attention, explicit interaction layers) or dataset size/feature count the current setup doesn't support.
