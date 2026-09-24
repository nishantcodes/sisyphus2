# Embedding MLP Baseline (Torch) — Same Engineered Features as XGB

**Change:** PyTorch MLP with learned embeddings for categoricals, fed the same affordability/charging/anxiety ratio features that lifted XGB to 0.9437, on the shared folds.

**Result:** CV 0.93858 ± 0.00098 — *worse* than every GBDT baseline, including the raw-feature ones (LGBM 0.9419, XGB 0.9415). The decorrelation hypothesis failed: error correlation with LGBM champion is 0.9792, prediction correlation 0.9852 — nearly the same model, not a different one.

**Why it didn't work (mechanism, not score):** The hypothesis was that smooth/additive latent-space functions would make different errors than axis-aligned tree splits. The error-analysis interactions found (Environmental_Concern_Level × Subsidy_Available, lift 2.4–2.8; Environmental_Concern_Level × Income bands, lift ~2.0) are sharp, threshold-like, categorical-crossed effects — exactly what trees are structurally suited to carve out and what a smooth MLP embedding space tends to blur/underfit. The MLP's own calibration is fine (deciles track closely) but its *capacity to isolate* these narrow high-lift categorical interaction cells is weaker than a tree's, so it converges to a similar-but-strictly-worse solution rather than a decorrelated one.

**Transferable takeaway:** Don't chase architecture diversity for blending value here — the missing signal is sharp categorical interactions (concern-level × subsidy, concern-level × income), not smooth numeric structure. Next lever: explicit interaction features (concern×subsidy, concern×income) fed back into the GBDT line, not further NN baselines.
