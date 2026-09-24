# Tabular modelling

- **GBDTs already model interactions.** Re-expressing a two-way interaction the
  error analysis flagged as an explicit feature usually adds nothing, because
  the trees found it already. Explicit features pay off for things trees cannot
  represent cheaply: ratios, differences, counts, aggregations across rows, and
  target statistics.
- **A slice with high loss is not automatically actionable.** High loss where
  the base rate is near 0.5 may simply be where the problem is genuinely
  uncertain. Ask whether the model is worse there than the data allows, not just
  whether loss is high.
- **Target/likelihood encoding must be fold-safe.** Compute it inside the fold
  loop on training rows only. Leakage here inflates CV and does not reproduce.
- **Seed-averaging reduces variance, not bias.** It tightens the fold spread; it
  will not move a model that is limited by its features.
