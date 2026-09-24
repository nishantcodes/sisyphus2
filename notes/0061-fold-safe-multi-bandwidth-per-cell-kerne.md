# Discovery: Multi-Bandwidth Kernel Smoothing → Neutral (No Lift)

**Change:** Replaced exact-value/band target encodings with fold-safe Nadaraya–Watson kernel smoothing on integer income and 0.1-km commute grids, computed per Concern×Subsidy cell, to capture fine-grained structure at three simultaneous resolutions.

**Result:** CV 0.94545 ± 0.00077 — baseline, no lift above exp21.

**Why it failed (mechanism):** The error analysis revealed the real bottleneck was not *fine structure within cells*, but the *magnitude of the Concern×Subsidy interaction itself*. Decile 7 gap (-0.5672) and the 1.60x loss lift on Concern=3 ∧ Subsidy=Yes (74.9k rows) show the model systematically under-predicts in that categorical region. Kernel smoothing encodes *continuous response gradients* but cannot directly express that a specific discrete cell combination needs a multiplicative correction. The calibration gap is too large and uniform within the cell for smoothing income/commute variation to matter — the cell's interaction weight itself is missing.

**Lesson for next attempt:** Interaction-based signal (target encoding, explicit Concern×Subsidy×Income cells) will outpace continuous smoothing when the biggest error concentrates in a categorical cell, not in how predictions vary along continuous axes within the cell.
