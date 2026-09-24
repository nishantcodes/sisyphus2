# Sparse Spline GLM Underperformed Due to Over-Regularization in Complex Regimes

## Result
CV dropped from 0.94947 (champion LGBM) to 0.94233 — a **0.71 percentage point loss**. The model made worse predictions overall, not better.

## Why It Failed
The hypothesis predicted that smoother log(duration) curves per macro period would outperform GBDTs in the 2009–10 regime (nr.employed 4963–5076, euribor 0.63–1.04, cons.conf −36 to −27). This region shows massive loss concentration: lift 3.1–3.4 on individual features, lift 4.17 in some two-way crosses (e.g., mid-range duration × low confidence).

However, **the error correlation was 0.9593** with the champion — both models fail on nearly identical rows. This reveals the core problem: the GLM's L2 shrinkage toward a global smooth curve is too conservative for this regime. The regime isn't just noisy; it has sharp, regime-specific boundaries (possibly tied to crisis psychology or policy changes) that smoothing erases. GBDTs capture these with small splits, accepting noise. The GLM over-regularizes, trading noise for bias.

## Takeaway
Smoothing isn't the missing piece; the regime needs **flexible, localized patterns that resist pooling toward the global trend**. Future attempts should either (1) use less regularization specifically in high-loss regions, (2) model regime indicators explicitly as separate features, or (3) pursue a genuinely different architecture (e.g., non-parametric methods without the shrinkage penalty).
