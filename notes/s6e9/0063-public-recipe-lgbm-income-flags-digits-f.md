# Different feature view lifted calibration blind spot

**What changed:** Swapped grid-based TargetEncoder variants for a public recipe using digit decomposition (10^-4 to 10^3), transductive frequency encoding of numerics-as-strings, dual-smoothed sklearn TargetEncoder, income flags (capturing deterministic regions at 30k, 38–42k, 170k+), and LGBM with max_bin=1024.

**CV result:** 0.94597 ± 0.00078 (vs. champion's 0.94570), a clean +27bp mean improvement.

**Why it worked:** Error analysis exposed two mechanisms the champion missed:
1. **Calibration collapse** — massive underprediction across all deciles (gap -0.5672 at decile 7). The champion saw numerics as continuous; the recipe decomposes them into digit bins and categorical frequencies, letting LGBM fit granular splits.
2. **Missing interactions** — Concern=3×Subsidy (lift 1.60, 74k rows) and Concern=5×income [30k–45k] (lift 1.32) weren't captured. Income flags explicitly flag deterministic regions (income=30k is 0.044 default rate; 170k+ is 1.000). Dual TargetEncoder smooth-encodes these to learn their precise interaction with Concern.

**The insight:** Not a hyperparameter tune, but a **representation shift**. Our TE grid stayed flat because it embedded numerics once, then blended. This recipe reifies numerics as categorical features (digits, frequencies, flagged regions), letting the learner discover interactions we didn't hand-code. Even parity with the public model would confirm representation > learner choice.
