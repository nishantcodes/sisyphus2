# s6e9 EDA — what the data actually is

Run by hand (not by the agent) on the full 668,665-row train / 286,571-row test.
Every number below is measured, and the script that produced it is reproducible.

## Shape

13 features, no `id` signal, target `Will_Buy_EV` at **17.46%** positive.
7 numeric, 6 categorical. **Zero missing values anywhere.**

## Three dead ends, closed by measurement

These are worth stating loudly because we spent experiments on two of them.

1. **No duplicate rows.** 0 exact duplicate feature-signatures in train, and
   **0 signatures shared between train and test**. Record linkage, near-duplicate
   matching, and "reconstruct the source record" have no purchase here. exp48
   built exactly this and landed at 0.9943 error-correlation with the champion —
   it reached the same predictions by a different road, which is what this
   measurement predicts.
2. **No distribution shift.** Adversarial validation (train vs test, 3-fold
   LGBM) gives **AUC 0.5001**. Train and test are drawn identically. So:
   adversarial reweighting, shift-corrected pseudo-labelling, and test-time
   distribution tricks are all inert. The upside is that **CV is trustworthy** —
   which matches our record of 5 of 6 submissions landing within 0.0002 of CV.
3. **No missingness signal.** There is none to model.

## The actual structure

`Environmental_Concern_Level` (5 levels) is the problem.

| level | target rate | n |
|---|---|---|
| 1 | 0.0057 | 147,476 |
| 2 | 0.0214 | 135,133 |
| 3 | 0.1109 | 127,351 |
| 4 | 0.2488 | 130,469 |
| 5 | 0.5183 | 128,236 |

Univariate AUC **0.8435** on its own. Marginal signal for every other feature:

| feature | marginal |
|---|---|
| Environmental_Concern_Level | 0.8435 |
| Annual_Income_USD | 0.6704 |
| Daily_Commute_km | 0.5338 |
| Charging_Stations_Near_Home | 0.5159 |
| Age / Number_of_Cars_Owned | ~0.505 |

**`Concern x Subsidy_Available` is a 10-cell crosstab that alone scores 0.91546.**
That is the single most important fact in this file. Every other feature pair in
the scan is just "concern plus noise" — the next best is 0.8515.

Subsidy is a **genuine interaction**, not an additive effect: its within-stratum
AUC rises monotonically with concern (L1 0.697 -> L5 0.759). Income, by contrast,
is **stable across strata** (0.669–0.687), so it is close to additive in logit
space — model it as a main effect, don't burn turns crossing it with concern.

## The generative fingerprint: round numbers

The synthesiser preserved a "round number" population from the original data.

| flag | n | rate inside | rate outside | diff |
|---|---|---|---|---|
| income % 1000 == 0 | 61,674 | 0.0443 | 0.1879 | **-0.1435** |
| income % 100 == 0 | 67,460 | 0.0539 | 0.1882 | -0.1343 |
| income % 10 == 0 | 133,245 | 0.1220 | 0.1877 | -0.0657 |
| commute on .0/.5 | 248,476 | 0.1767 | 0.1734 | +0.0033 |

Round-income rows are **4x less likely** to buy. Income occupies only 8.3% of its
integer grid (13,214 distinct values over 30,000–188,549) and is ~2x enriched on
multiples of 5, so this is structural, not chance.

Adding explicit modular flags + trailing-digit values to the raw features is
worth **+0.0015** on a 3-fold reference (0.940614 -> 0.942107) — the same order as
the two best gains of the whole campaign. exp39's decimal-digit features capture
part of this already (it holds the 2nd-highest blend weight), but the explicit
`% 1000` / `% 100` flags are a cleaner encoding of the same fact.

## The ceiling ladder

| features | 10-fold CV |
|---|---|
| concern only | 0.842790 |
| concern + income | 0.880960 |
| all 13 raw | 0.941515 |
| **our best single model (exp21, verified)** | **0.945476** |
| best blend (exp36, 21 members) | 0.945700 |

Raw features get 0.9415. Twenty-six experiments of feature engineering have
bought **+0.004** on top of that. Every model family we have tried — GBDTs, MLP,
sparse GLM, record-linkage — lands at 0.968–0.998 error correlation with the
champion. This problem is close to its Bayes limit and the remaining headroom is
in **representation**, not architecture.

## Where the remaining signal must be

If concern x subsidy (10 cells) plus income gets most of the way, then whatever
is left lives in the **fine structure of income and commute inside each of those
10 cells**. That is the place to spend turns.

Candidates that follow from the measurements and are NOT yet tried:
* Per-cell income modelling — splines, per-cell quantile bins, or a monotone
  income effect fitted separately within each of the 10 concern x subsidy cells.
* Explicit modular flags (`%1000`, `%100`, `%10`, `%5`) and trailing-digit
  residuals as first-class features rather than derived digits.
* Dropping the near-noise features (Gender 0.0054, Current_Car_Type 0.0245,
  City_Type 0.0323 marginal spread; Age ~0.505 everywhere). Fewer columns is
  less to overfit at a noise floor this tight.
* ~~Merging the original (non-synthetic) EV dataset~~ — **tested 2026-09-03 and
  rejected.** The source is `itzzomkar/ev-adoption-behavior-and-range-anxiety`
  (10,000 rows, schema confirmed). Zero of its rows match train or test, and
  adding it degrades CV monotonically in sample weight (-0.000024 / -0.000190 /
  -0.000717 at weight 1/3/10). The synthesiser compressed the tails and imputed
  the nulls, so the original is off-distribution. See notes/eda-s6e9.md.
