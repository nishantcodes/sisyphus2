# FM on Discrete Tokens — Interaction Modeling Without the Capacity

**Change:** 2nd-order factorization machine trained on discretized income/commute tokens (digits, exact values, flags) instead of continuous features.

**Result:** CV 0.94380 ± 0.00083, underperforming champion (0.94597 ± 0.00078) by ~0.0022.

**Why it failed:** The FM's core strength — learning pairwise interactions as rank-constrained bilinear forms — proved insufficient for the true interaction structure. The error analysis pinpoints exactly where the champion struggles: Concern×Subsidy (lift 2.45–2.80) and Concern×fine-income-band interactions (lift 2.27–2.14). These are not low-rank; they're sharp, discontinuous boundaries that LGBM's axis-aligned splits capture naturally but FM cannot represent without explicit high-rank tensors. 

The discretization strategy (tokens instead of TE-encoded continuous values) also cost expressiveness: the champion's target-encoded income bins carry soft continuous information; FM's discrete tokens force hard boundaries, losing the smooth gradient LGBM learns within each bin.

**Transferable insight:** Interaction-aware embeddings (FM's learned tokens) are not a substitute for interaction-discovering splits (LGBM). For this domain, the interaction surface is too jagged and label-dependent for FM's smooth bilinear form. Next: either increase FM's rank (high regularization cost) or accept that tree-based TE + splits outmatch tensor methods here.
