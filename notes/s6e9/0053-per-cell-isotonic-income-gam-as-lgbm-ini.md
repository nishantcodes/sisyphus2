# Per-Cell Income Isotonic GAM as Init Offset — No Lift

**What changed:** Replaced standard income features with a fold-safe isotonic (monotone) income response fitted separately for each of 10 Concern×Subsidy cells, converted to logits, and passed to LightGBM as an `init_score` offset. Also encoded the income artifact explicitly (income % 1000 == 0 buys at 0.044 vs 0.188).

**Result:** CV 0.94508 ± 0.00076 — no improvement vs champion (0.94570).

**Why it didn't work:**

The error analysis showed excess loss concentrated in Concern=3×Subsidy=Yes (lift 1.60) and Concern=5 narrow income bands (lift 1.17–1.32), suggesting per-cell income stratification should help. **But:**

1. **Monotonicity was too restrictive.** Even within a cell, the income-to-purchase relationship may be non-monotonic (e.g., different behavior at round-number income thresholds); isotonic removes that flexibility entirely.

2. **Init-score offset doesn't compose well with LGBM's splits.** The booster can't easily "work around" a stratified offset—it still has to learn residuals, and if those residuals have structure (other feature interactions, income artifacts), the model gets stuck optimizing around a bad initial guess rather than ignoring it.

3. **Interaction was deeper than income alone.** The problem cells are defined by *both* Concern and Subsidy; within those cells, income's signal may be entangled with other unmeasured factors (dealer behavior, incentive timing) that a purely income-stratified prior can't untangle.

**Takeaway:** Stratified priors via offset work only if the offset captures the *dominant* signal in each stratum. Here, income was a minor component. Better path: explicit Concern×Subsidy×Income features or a local model blend within problem cells.
