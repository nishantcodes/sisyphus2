# Discovery: Day-Level Features Capture Temporal Variation in Campaign Success

## Change
Added two day-level engineered features: (1) per-day call volume, and (2) nested fold-safe day-key success rate (smoothed).

## Result
**CV improved: 0.95039 ± 0.00287** (from baseline).

## Why It Worked: The Mechanism

The raw data (41,188 rows, identical to original UCI dataset) maps almost 1:1 to individual calendar days—one unique (month, day_of_week, euribor3m, cons.conf.idx, emp.var.rate) tuple per day. Success rates shift sharply between small campaign batches.

The error analysis revealed the champion's blind spot: 3–4× excess loss in economically sensitive subregions (duration × low employment, duration × low euribor, duration × low confidence). These regions concentrate in the post-2008 low-employment regime where campaign success is highly volatile and day-dependent.

Trees cannot cheaply isolate hundreds of distinct calendar days as separate nodes. By directly injecting day-level features (call volume and smoothed success rate), the model bypassed expensive feature engineering and gained immediate access to the day-level signal that error analysis showed was critical. This reduced the model's reliance on expensive interaction trees in the volatile regime.

## Key Insight
**Temporal aggregation at the grain that matches your data structure works.** When features naturally cluster to a higher-level entity (here, calendar days), explicitly modeling that level outperforms hoping trees will find it through interactions.
