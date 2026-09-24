# Exp #9: Day-relative features unlock duration slices

## Change
Added target-free group aggregates derived from day-of-call (macro tuple × month × day_of_week): group sizes, mean/std duration, and each row's duration and campaign rank *within its day*. These capture relative position in the daily call volume that raw duration alone cannot expose.

## Result
CV improved to **0.95135** (AUC 0.95101, mean loss 0.1102) from prior baseline.

## Why it worked
Error analysis identified the champion's worst region: long-duration calls (lift 3.91), especially when crossed with low nr.employed (lift 3.90). The model couldn't distinguish these high-duration outliers from the mass.

The insight: a decision tree must reconstruct "this call was long *for today*" by iteratively splitting raw duration against the distribution in small subsets. Providing within-day rank directly exposes what the tree must learn, bypassing the synthetic complexity.

This was especially valuable because Exp #5 (GAM, structurally different) made nearly identical errors (correlation 0.9419), suggesting the bottleneck was feature-space, not model flexibility.

## Transferable takeaway
When trees underperform in high-value regions that involve relative position or concentration, adding group-relative aggregates (rank, percentile, z-score) often outperforms adding raw aggregates. The tree still needs to learn *when* to use them, but the translation cost drops.
