# Sparse One-Hot Logistic Regression + Pairwise/3-Way Crosses: CV 0.94099 (–0.0047 vs champion)

## Change
Replaced tree-based ensemble with a sparse sklearn logistic regression on one-hot categorical features + explicit pairwise and three-way interaction terms, L2 regularization, no target encoding.

## What Happened
**Baseline performance, no lift.** This scored +0.0047 *below* the current champion (0.94570), confirming trees remain the dominant model family here.

## Why It Failed: The Mechanism

The error analysis showed the hypothesis was *half right*: categorical interaction cells like Concern=3 × Subsidy=Yes do carry 1.61× excess loss, concentrated and real. But the fix assumed linear cross terms could capture this signal.

**The actual structure is hierarchical, not additive.** Trees learn *conditional* decision boundaries (e.g., "when Concern=3 AND Subsidy=Yes, apply nested splits on Income"), while explicit one-hot crosses only model flat pairwise sums. The severe miscalibration (decile 1–7 predictions 10–500× actual rates) exposed that a linear model's probability surface fundamentally mismatches the data's non-linear structure.

Further evidence: exp #42 (adding fold-safe exact Concern×Subsidy targeting to a tree) has –0.0024 error correlation with the champion — they find *different* signal in that same interaction cell. This confirms the signal exists but requires hierarchical modeling to unlock, not linear terms.

**Transferable lesson:** Explicit categorical crosses fix *location* but not *mechanism*. When error is concentrated in specific cells, trees will find it; linear models need the trees to teach them the decision boundaries first (via TE or stacking).
