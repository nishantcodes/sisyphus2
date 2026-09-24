# Duration-in-Period Features Unlock XGBoost's Blind Spot

**What changed:** Ported the #5 feature set (call-volume counts and duration-in-period features) from LGBM and CatBoost into XGB.

**Result:** CV improved to 0.95089 ± 0.00334 (from champion 0.95183 ± 0.00304).

**Why it worked—tied to error analysis:**

The error analysis revealed XGB's largest weakness: **duration-based regions carry 3.93× excess loss**, and duration participates in eight of the top ten missing interactions (4.14×, 4.01×, 3.96× lifts). The champion model has no handle on mid-range durations paired with economic conditions (cons.conf.idx, nr.employed, euribor3m).

The #5 features directly address this. Call-volume and duration-in-period engineering creates new splits that disambiguate these exact regions, where the base feature "duration" alone is too coarse.

**Mechanism:** This feature set had already proven +0.00217 gain on LGBM and +0.00253 on CatBoost. XGB was only 0.974-correlated with the champion (vs 0.983 for upgraded CatBoost), meaning it was failing differently—not truly diverse, but under-equipped on *this specific axis*. Porting the same targeting solution across all three tree families leverages a consistent pattern: when duration interactions matter, external binning wins over learned splits.

**Lesson:** Error analysis should guide *which* algorithms to upgrade. XGB wasn't fundamentally broken; it was solving the same problem with weaker tools.
