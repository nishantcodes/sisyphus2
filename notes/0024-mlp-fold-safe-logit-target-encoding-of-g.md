# MLP + Fold-Safe Logit Target Encoding of Grid/Concern Cells

**Change:** Fed the same fold-safe target-encoded grid/interaction cells (Concern×Subsidy, Concern×income-bin) that fixed the GBDTs into the embedding MLP, as pre-computed logits so the network only needs a linear combination rather than learning the cell statistics itself.

**Result:** CV 0.94378 ± 0.00074, up from the MLP baseline's 0.93858 — a +0.0052 jump, the largest single-change gain in the campaign, though still below the GBDT-family ceiling (~0.9453-0.9455).

**Why it worked (mechanism, not score):** The MLP baseline's error analysis showed the exact same high-lift cells as the pre-TE GBDTs (Concern=4/5 × Subsidy=Yes, lift 2.45–2.82; Concern × narrow income bands, lift ~2.0–2.2) — proof the MLP's weakness wasn't its architecture but its inability to represent sharp per-cell base rates from smooth embeddings. Passing those rates in as logits removes the representation burden entirely; the net just has to weight a pre-solved signal instead of approximating it through embedding geometry.

**Why it still trails GBDTs:** Error correlation with CatBoost is still high (0.9827) — the TE features now dominate the fit for *any* learner, so the MLP converges toward the same solution but doesn't surpass trees, which remain better at capturing whatever residual structure (other undiscovered crosses) TE doesn't cover.

**Transferable takeaway:** A representation fix that closes a learner's blind spot doesn't automatically make it a better model than learners that already had partial access to that structure — check post-fix error correlation, not just CV, before counting on it for blend diversity.
