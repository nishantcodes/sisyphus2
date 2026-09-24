# LGBM Baseline — Native Categoricals, Shared 5-Fold CV

**Change:** LightGBM on raw features, native categorical handling (no encoding), 5-fold shared CV splits.

**Result:** CV 0.94185 ± 0.00093. First scored model in this family — establishes the reference point.

**Why this was the right first move:** No leakage-safe baseline existed yet, so there was nothing to compare future model families against and no OOF/PRED arrays to mine for error structure. Using native categorical support (rather than one-hot/target encoding) avoids introducing fold-leakage risk at this stage and keeps the baseline "honest" — the score reflects raw signal in the data, not encoding cleverness.

**Why it likely worked (mechanism, not just score):** LightGBM's histogram-based categorical splits handle high-cardinality fields (city type, charging availability tiers) without exploding dimensionality the way one-hot would, so the model can find interactions among income bands / charging availability / city type natively — exactly the segments error analysis flagged as likely failure zones.

**Transferable takeaway for next turn:** Use the saved OOF/PRED arrays to slice errors by income band, charging availability, and city type. Low variance (±0.00093) suggests folds are stable — any future gain should come from feature engineering or target encoding on the categorical splits this baseline didn't explicitly engineer, not from fold instability.
