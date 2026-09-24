# LGBM Baseline: Clean Reference with Native Categoricals

## What Changed
Trained a LightGBM baseline on raw features using native categorical handling and correctly-stratified shared folds, with OOF and test predictions saved for downstream error analysis.

## Score & Findings
**CV: 0.94978 ± 0.00289** — strong baseline performance with tight variance across folds.

## Why This Approach Works

The key insight was **data provenance**: row counts (41,188) exactly match UCI bank-additional-full, proving train/test are splits of real data, not synthetic duplicates. This rules out a tempting shortcut: adding the original full dataset would leak test information. 

Instead, this baseline takes a defensible path: use only the training fold with honest cross-validation, establish native categorical support in LightGBM (avoiding one-hot encoding overhead), and freeze predictions for inspection. The tight CV variance signals the folds stratify well and the model is stable.

## Value for Next Steps
The saved OOF predictions enable proper error analysis—identifying which samples or feature patterns drive misclassification—without the circular reasoning of looking at test performance first. This reference frame lets future iterations target weaknesses with evidence rather than guessing.

The deliberate omission of extra data here is actually a strength: it keeps the experimental scope clean and prevents masking real model improvements with leakage gains.
