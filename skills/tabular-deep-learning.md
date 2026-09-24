# Modern tabular models (verified available in this environment)

`pytabkit` is installed and gives pre-tuned, sklearn-style implementations of
the current tabular-DL literature. These are NOT exotic — TabM is the reported
best DL baseline in its paper, and RealMLP is the model family that won a
recent Kaggle playground competition as a *single model*.

    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier, \
                         TabR_S_D_Classifier, FTT_D_Classifier, \
                         MLP_PLR_D_Classifier, XRFM_D_Classifier

**The exact class list — import nothing outside it.** An experiment already
failed by inventing `RealMLP_PLR_TD_Classifier`. Every name below takes the
suffix `_Classifier`:

CatBoost_D, CatBoost_HPO, CatBoost_HPO_TPE, CatBoost_TD, Ensemble_HPO,
Ensemble_TD, FTT_D, FTT_HPO, LGBM_D, LGBM_HPO, LGBM_HPO_TPE, LGBM_TD,
MLP_PLR_D, MLP_PLR_HPO, MLP_RTDL_D, MLP_RTDL_HPO, MLP_SKL_D, RF_HPO, RF_SKL_D,
RealMLP_HPO, RealMLP_TD, RealMLP_TD_S, RealTabR_D, Resnet_RTDL_D,
Resnet_RTDL_HPO, TabM_D, TabM_HPO, TabR_HPO, TabR_S_D, XGB_D, XGB_HPO,
XGB_HPO_TPE, XGB_PBB_D, XGB_TD, XRFM_D, XRFM_HPO

PLR numeric embeddings are `MLP_PLR_D_Classifier`, not a RealMLP variant.

The useful families:

| model | what it is |
| --- | --- |
| `RealMLP_TD_Classifier` | strongly pre-tuned MLP; best-in-class default MLP |
| `TabM_D_Classifier` | MLP + parameter-efficient (BatchEnsemble) ensembling |
| `TabR_S_D_Classifier` | retrieval/kNN-augmented deep model |
| `FTT_D_Classifier` | FT-Transformer — feature tokens + attention |
| `MLP_PLR_D_Classifier` | MLP with periodic (PLR) numeric embeddings |
| `XRFM_D_Classifier` | recursive feature machine with adaptive tree structure |

`_TD_` = tuned defaults (no search needed), `_D_` = defaults, `_HPO_` = runs a
search (expensive — do not use on the full frame here).

## Gotchas measured on THIS data, not guessed

* **Categoricals must be integer codes.** `pandas` reads this competition's
  string columns as the arrow-backed `str` dtype, so `df[c].dtype == object` is
  FALSE and a dtype check written that way silently misses them. Use
  `pd.api.types.is_numeric_dtype(df[c])`, then `pd.Categorical(df[c]).codes`,
  then cast the frame to float32 and pass `cat_col_names=[...]`.
* **Not every model takes every kwarg.** `use_ls=False` is right for RealMLP on
  AUC/log-loss, but `TabM_D_Classifier` raises `TypeError` on it. Construct with
  the minimum and add kwargs only where the model documents them.
* **pytabkit defaults to CPU even when MPS exists** — it prints "GPU available:
  True (mps), used: False". Pass `device=os.environ.get("SIS_DEVICE","cpu")`.
* **Timing, measured:** `RealMLP_TD_S_Classifier` took 43s for ONE fold on 30k
  rows on CPU. Full data (668k rows x 10 folds) extrapolates to ~2.7 HOURS,
  which exceeds the local timeout. Either run these on a Kaggle kernel, use the
  `_S_` (small) variants, cut folds, or subsample. Do not casually fit one of
  these on the full frame locally.

## Optimisations that matter for a ranking metric (AUC)

* **Turn label smoothing OFF** (`use_ls=False`) — it is on by default in some
  pre-tuned configs and hurts AUC/log-loss.
* **Post-hoc calibration**: pytabkit supports `calibration_method='ts-mix'`
  (temperature scaling with Laplace smoothing) and refinement-based early
  stopping via `val_metric_name='ref-ll-ts'`. Calibration does not change AUC
  by itself (AUC is rank-only) but it does change how a model BLENDS with
  others, so it matters for ensembling.
* **`n_ens` / PackedEnsemble**: RealMLP accepts `n_ens > 1` to train several
  heads per split. Cheaper than separate seeds for the same variance reduction.
* **Rank-transform before blending.** Different model families are calibrated
  differently; averaging raw probabilities lets a confidently-wrong model
  dominate. `rankdata(p)/len(p)` first, then optimise weights on the metric.

## Which of these to reach for

The measured problem on this dataset is that every model tried so far — GBDTs,
an MLP, and a sparse linear GLM with crosses — has error correlation 0.968-0.998
with the champion. Architecture changes have NOT produced decorrelation. So
prefer a model here for its own accuracy, not on the hope that it will
decorrelate, and verify decorrelation empirically rather than assuming it.
