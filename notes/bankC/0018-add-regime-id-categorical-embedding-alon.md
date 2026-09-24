# Regime Embedding Addition: Why Duplication Underperformed

## Change
Added categorical regime_id embedding alongside (not replacing) periodic encoding of all 10 numeric columns, preserving fine-grained continuous signal (euribor3m has 309 distinct values) while adding discrete regime membership information.

## Result
CV degraded from 0.95028 to 0.94453 (–57 bps), contrary to hypothesis.

## Why It Failed
The hypothesis correctly identified that #16's regime-only approach threw away signal by replacing continuous features. However, **adding** regime_id didn't recover that loss—it worsened performance. Three likely mechanisms:

1. **Feature redundancy without benefit**: The categorical regime_id doesn't partition the decision boundary sharply enough. Error analysis shows loss concentrations (duration×cons.conf.idx interactions, euribor3m slices) where regime membership doesn't cleanly segment risk. LGBM's #9 success came from *treating* regime_id as a categorical split point, not as dual representation.

2. **Embedding mismatch**: A learned categorical embedding may not align with how macro columns actually cluster risk. The regime bucketing is coarse by design; mirroring it in embedding space adds noise rather than complementary signal.

3. **Model capacity constraint**: Neural network split-learning was already saturated at #9's performance. Adding a correlated feature wastes capacity on redundancy instead of learning novel interactions the error analysis reveals are missing (duration×euribor3m, duration×nr.employed).

**Lesson**: Ensure new features partition unmodeled error, not replicate existing structure.
