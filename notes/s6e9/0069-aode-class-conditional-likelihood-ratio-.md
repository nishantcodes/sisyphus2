# AODE Class-Conditional Bayes Net — Failed Joint-Distribution Hypothesis

**Change:** Tested AODE (averaged one-dependence Bayes net using discretized likelihood ratios) to model P(x|y) directly, specifically targeting missing interactions flagged in error analysis: Concern×Subsidy (lift 2.80, n=80k) and Concern×Income (lift 2.27–2.14, n=13–14k each).

**Result:** CV 0.94057 ± 0.00080 — **baseline, 54 points below champion** (LGBM 0.94597). Hypothesis failed.

**Why it likely failed — mechanism:**

Error analysis showed the champion's worst regions are *fine-grained interaction cells*, not coarse feature pairs. AODE's discretized histogram-ratio approach (15 averaged one-dependence models over binned income and concern) cannot resolve sub-bin structure — Concern=5 × Income∈(78.9k, 84.9k] is a single cell treated as uniform, but the champion's continuous splits capture gradients within it. 

The "missing interaction" lifts are actually a *failure of discrete binning*, not a model architecture problem. A generative classifier that discretizes income into 10 bins inherits that ceiling; blending it with the champion adds only noise (0.7831 prediction correlation, confirmed by poor disagree-set performance).

**Lesson:** When error analysis flags "missing interaction" in a continuous domain, the fix is feature engineering (poly terms, target encoding) or finer splits, not switching to generative + discrete. Continuous discriminative models already capture fine-grained interaction structure via splits; the gap here was feature *availability*, not model form.
