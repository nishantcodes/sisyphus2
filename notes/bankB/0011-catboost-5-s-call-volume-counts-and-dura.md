# CatBoost + Volume Features: Decorrelated but Structurally Weaker

**Change:** Ported call-volume counts and duration-in-period features from exp #5 (LGBM, CV 0.95164) into CatBoost, expecting to lift CatBoost's standalone 0.9496 closer to the champion.

**Result:** CV 0.95077 — **worse than champion by 0.00087**, despite features proven to add +0.00217 in LGBM.

**Why it failed — the mechanism:**

CatBoost's ordered boosting and oblivious tree structure are misaligned with how these features need to be used. Error correlation (0.969) is high, meaning CatBoost and LGBM fail on the same rows despite different architectures. The volume features address known loss concentrations (lift 3.16 in nr.employed band, 4.1× in duration × cons.conf.idx), but CatBoost couldn't extract that signal.

The problem: **feature sufficiency ≠ model sufficiency.** Adding X to a weaker model doesn't close a structural gap. LGBM's histogramming and leaf-wise growth capture duration-volume interactions CatBoost's trees miss. High decorrelation requires different *failure modes*, not just different architectures. Here, both fail in the 2009+ regime; giving CatBoost identical features doesn't fix the regime itself.

**Lesson for blending:** Don't assume a decorrelated model + proven features → complementary predictions. Validate that decorrelation comes from *different handling of the target signal*, not just noise orthogonality.
