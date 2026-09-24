# Mixture-of-Experts by Cell Failed (exp#XX)

**Changed:** Split into 10 LGBM experts per Concern×Subsidy cell; tested income fingerprint (income%1000==0); dropped low-signal columns (Gender, Age, Current_Car_Type).

**Result:** CV 0.94425 (down 0.0115 vs champion's 0.94570).

**Why it failed:** Error analysis reveals the champion's power isn't per-cell specialization—it's *cross-cell decorrelation*. The champion (21-model blend) achieves near-zero error correlation (0.0012) with exp#42, meaning different global models catch different hard regions. Concern=3×Subsidy=Yes is the worst interaction (lift 1.60), but Subsidy_Available=Yes has lift 1.22 across all 420k rows, signaling that subsidy×income/commute interactions live *globally*, not confined within cells.

Cell-based experts fragmented this: each expert viewed subsidy only through its own cell's income distribution, severing the global calibration that the ensemble blend provides. The income fingerprint (+0.0015 isolated) was too small to recover the loss.

**Lesson:** When error correlation is near zero, the ensemble's strength is disagree-and-recover on hard regions, not specialization. Mixture-of-experts pays off only if regions of high loss stay isolated per partition. Here they spilled across partitions, so fragmentation destroyed ensemble synergy.
