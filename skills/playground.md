# Kaggle Playground Series

Playground competitions are **synthetic** — generated from a real-world dataset.
That single fact drives most of what wins them.

- **Find and add the original dataset.** The generator's source data is almost
  always a public dataset (UCI, an older Kaggle dataset) and is usually named in
  the competition discussions within days. Concatenating it to the training data
  is frequently the largest single gain available in the whole competition, and
  no amount of feature engineering substitutes for it. If you have not checked
  for it, that is the highest-value thing you can do.
- **The leaderboard is compressed.** Median to first place is often under 0.01
  AUC. Gains of 0.001 are meaningful, the fold spread matters enormously, and
  more folds buy real discrimination between candidates.
- **Synthetic artifacts are exploitable.** Generated columns are often
  discretised or rounded. Rounding a float back to its underlying grid, or
  treating a low-cardinality float as categorical, sometimes helps.
- **Public kernels converge fast.** Within a week most of the field runs a
  similar blend. Beating the median means doing something it does not.
- **Ensembling is usually expected — but verify it on THIS data.** Measure error
  correlation between your models before spending turns on diversity. If every
  pair is above ~0.97 and blends score below the best member, ensembling is a
  dead end here and the gain has to come from a stronger single model.
