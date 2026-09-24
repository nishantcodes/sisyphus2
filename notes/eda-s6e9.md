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
* **Merging the original (non-synthetic) EV dataset** — standard playground
  practice, and we have never done it. There is no `original.csv` in `input/`.

## The original dataset: TESTED AND REJECTED (2026-09-03)

`itzzomkar/ev-adoption-behavior-and-range-anxiety` (10,000 rows) is confirmed to
be the source dataset for this competition: every competition column present,
all categorical levels identical, target rate 0.1750 vs 0.1746.

**Do not propose merging it. It was measured and it hurts.**

* Zero leakage: **0 of 10,000** original rows match a train or test row exactly.
  The synthesiser regenerated every row, so there is nothing to look up.
* Adding it as extra training rows degrades CV, and does so **monotonically in
  the sample weight** — weight 1: -0.000024, weight 3: -0.000190, weight 10:
  -0.000717 (3-fold LGBM on raw features). A noise effect would change sign;
  this does not.
* Why: the synthesiser **compressed the tails** (original income reaches 223,300
  vs 188,549 in train; commute 135.5 vs 98.7) and imputed away nulls (the
  original has 178/181/184 missing in income/commute/concern, the competition
  data has none). The original is off-distribution, and with adversarial
  AUC 0.5001 between train and test, off-distribution rows are contamination.

The general lesson for playground competitions: "merge the original dataset" is
a *hypothesis to test*, not a rule. Check the tails and the null pattern first —
if the synthesiser changed them, the original is a different distribution.

## CORRECTIONS + what a public 0.94587 single model does that we do not (2026-09-03)

Public kernel `najiama/pure-lgbm-model-cv-0-94587-lb-0-94612` is a **single LGBM**
at CV 0.94587 / LB 0.94612. Our best single model is CV 0.945476 / LB 0.94563.
Everything below is verified against our own train.csv.

### Two corrections to earlier notes in this file

1. **The "round number population" was mostly ONE artifact.** 61,605 rows sit at
   *exactly* `Annual_Income_USD == 30000` — a mode-collapse spike where the
   synthesiser clipped the floor — and they are 61,605 of the 61,674 rows in the
   `income % 1000 == 0` finding. Encode `income == 30000` as its own flag; do not
   reason about general "roundness".
2. **Rejecting the original dataset was a rejection of ONE usage.** Appending its
   rows as extra training data does hurt (measured, monotone in weight). But the
   0.94587 kernel uses it a completely different way and we have NOT tested that:
   for every column, map value -> target mean computed **on the original
   dataset**, and feed that as a feature (`{col}_org_mean`, filled with the
   original's global mean). This injects real-world prior signal without adding
   off-distribution rows. **This is untested by us and belongs in a rival's hands.**

### Deterministic income regions we never found

| region | n | buy rate |
|---|---|---|
| `income >= 170000` | 393 | **1.0000** |
| `income == 30000` | 61,605 | 0.0443 |
| `38000 <= income <= 42000` | 1,277 | 0.0016 |
| global | 668,665 | 0.1746 |

A 393-row region at a 100% rate is pure signal that a depth-limited tree will not
isolate on its own. Encode these as explicit boolean flags.

### The rest of that kernel's recipe (all absent from our experiments)

* **Digit features over a wide range**: `for k in range(-4, 4)`:
  `(col.fillna(0) // 10**k % 10).astype('int8')` on every numeric column,
  including the 10^-4..10^-1 decimal places.
* **Numeric -> string -> category**, then **global frequency encoding computed on
  train AND test concatenated** (transductive; legitimate, test labels unused).
* **Dual sklearn `TargetEncoder`** on all of those columns, fitted twice with
  different smoothing (`smooth='auto'` and `smooth=10.0`), both kept as features.
  Uses `TargetEncoder(shuffle=True, cv=5, random_state=42)` internal CV.
* **Drops `Number_of_Cars_Owned`** — matches our own finding that it is noise
  (marginal AUC 0.5039).
* **Drops perfectly-correlated (|r| == 1.0) and constant columns** automatically.
* **LGBM hyperparameters**, very different from ours: `n_estimators=40000`,
  `learning_rate=0.005`, `max_depth=5`, `num_leaves=32`, `min_child_samples=10`,
  `subsample=0.8`, `colsample_bytree=0.3`, `reg_alpha=0.071`, `reg_lambda=2.0`,
  **`max_bin=1024`**, `early_stopping(700)`. Low LR + huge tree budget + a much
  finer histogram than the default 255 bins.

## pytabkit: the EXACT importable class names

exp57 failed by inventing `RealMLP_PLR_TD_Classifier`, which does not exist. The
complete list of classifiers — import nothing outside it:

CatBoost_D, CatBoost_HPO, CatBoost_HPO_TPE, CatBoost_TD, Ensemble_HPO,
Ensemble_TD, FTT_D, FTT_HPO, LGBM_D, LGBM_HPO, LGBM_HPO_TPE, LGBM_TD,
MLP_PLR_D, MLP_PLR_HPO, MLP_RTDL_D, MLP_RTDL_HPO, MLP_SKL_D, RF_HPO, RF_SKL_D,
RealMLP_HPO, RealMLP_TD, RealMLP_TD_S, RealTabR_D, Resnet_RTDL_D,
Resnet_RTDL_HPO, TabM_D, TabM_HPO, TabR_HPO, TabR_S_D, XGB_D, XGB_HPO,
XGB_HPO_TPE, XGB_PBB_D, XGB_TD, XRFM_D, XRFM_HPO

— each with the suffix `_Classifier`. PLR embeddings are `MLP_PLR_D_Classifier`,
NOT a RealMLP variant.
