# Discovery: Call-Volume Periodicity Captures Missing Calendar Signal

**Change:** Added call-volume counts (number of calls per macro period and day) and duration-relative-to-period as features to LightGBM.

**Result:** CV improved to 0.95164 (+0.00042), driven by better fit in the 2009+ low-employment regime where the previous model had 3-4x loss concentration.

**Why it worked:** The error analysis revealed the model was systematically underfitting `nr.employed ∈ (4963, 5076)` + `cons.conf.idx ∈ (-36.1, -26.9)` interactions, especially in mid-long-duration cells (lift 3.9–4.1). These macro slices pin down calendar months, but within months the conversion rate swings wildly. Trees cannot natively compute "how many instances share this value in the training set," so they miss that call volume is strongly predictive: low-volume periods convert far better in the real bank data. The `id` column is shuffled (Spearman ρ = 0.004 with `nr.employed`), carrying no temporal signal. Adding call counts per period/day and duration relative to period allows the model to discover this calendar + volume effect and fit the previously opaque interactions.
