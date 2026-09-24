# Interaction-Constrained Boosting Hit the Ensemble Ceiling

**What changed:** Swapped unconstrained LGBM for GA2M-style interaction-constrained variant: every tree could only split on {Concern, Subsidy} plus ONE other feature family, enforcing the detected stratified structure.

**CV result:** Reverted to baseline 0.94467 ± 0.00078 (−10 points vs. champion blend 0.94570).

**Why it failed — the mechanism:** The error analysis correctly diagnosed the problem: only ONE genuine interaction exists (Concern=3×Subsidy=Yes, 1.60× loss lift), plus within-strata modulation by income (1.17–1.32×). The hypothesis was sound — unconstrained boosting wastes capacity on spurious noise-column interactions, hurting generalization at this tight Bayes-limit regime.

But the constraint was *too rigid*. Hard GA2M enforcement is brittle: it forbids the model from exploring weaker but real sub-interactions (e.g., Concern=5×income-band nuance) and forces binning at the constraint boundary. The 21-model ensemble that achieved 0.94570 learned these structures implicitly through diverse weak learners, each overfitting slightly differently — ensemble averaging is more forgiving than architectural constraint.

**Transferable finding:** When error analysis pinpoints a structure, validate with soft regularization (interaction penalties, shrinkage, sparse trees) *before* hard structural constraints. Hard constraints traded interpretability for rigidity at the noise floor.
