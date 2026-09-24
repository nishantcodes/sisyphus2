# Discovery Note: Fold-Safe Target Encoding Fixes CatBoost's Structural Blind Spots

**Change:** Ported fold-safe target encoding (Environmental_Concern_Level × Subsidy_Available, Concern × Annual_Income_USD grid bands) to CatBoost with GPU training + early stopping. Prior attempts (#22 bug, #25 CPU timeout) failed to deliver the representation fix.

**Result:** CV 0.94550 ± 0.00076 (vs #19 champion 0.94179 ± 0.00090), a +0.0037 lift matching the predicted ~0.0035 gain.

**Why it worked (mechanism):** The error analysis of #19 revealed two high-lift interaction cells that CatBoost could not capture natively:
1. **Concern=4/5 × Subsidy=Yes**: Lift 2.45–2.81 on 80–95k rows — 42% of Subsidy=Yes population
2. **Concern=4/5 × narrow Income bands**: Lift 2.0–2.24 on 12–14k-row cells

These are *ordinal × binary* and *ordinal × binned continuous* interactions where CatBoost's ordered boosting + native categorical handling still fails to split effectively without explicit target statistics. The fold-safe TE injects the exact conditional probability `P(target | Concern, Subsidy)` and `P(target | Concern, Income_bin)` as dense numeric features, giving the gradient booster a direct signal instead of requiring it to discover deep interaction paths.

**Evidence:** On the 1% disagreement rows with LGBM #21 (which already had the TE fix), #19 lost 0.497 vs 0.298 — a 67% higher loss precisely where the missing interactions live. The fix closes this gap.

**Transferable insight:** Even "categorical-native" boosters (CatBoost) benefit from fold-safe target encoding on high-cardinality ordinal × categorical/continuous interactions when the interaction cells are large (>10k rows) but sparse in the split space. The TE features act as *learned interaction prototypes* that the booster then refines.
