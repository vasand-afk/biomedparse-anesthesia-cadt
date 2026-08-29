# Intraoperative Hypotension Prediction from Arterial Waveforms

Predicts a hypotensive event (MAP < 65 mmHg, sustained) 5 minutes ahead
from a 20-second raw arterial-waveform window, matching STEP-OP's real
published design (Lee et al., JMIR Medical Informatics 2021, PMC8517810,
Seoul National University) -- a 1D CNN + LSTM ensemble on the same
dataset this project uses.

## Data

**VitalDB** ([vitaldb.net](https://vitaldb.net/), free API access, no
credentialing gate) -- 3,644 real surgical cases with both `SNUADC/ART`
(raw 500Hz arterial waveform) and `Solar8000/ART_MBP` (numeric mean
arterial pressure) tracks. Preprocessed to 2,958 usable cases with real
hypotension events + matched negative windows.

## Real, documented bugs found and fixed during development

1. **NaN fragmentation of every real hypotensive episode**: `ART_MBP`
   only updates every ~2 seconds at the device level, so naive 1Hz
   loading treats every other sample as missing. Treating missing as
   "not hypotensive" fragmented every real sustained episode into
   isolated 1-sample blips, returning zero detected events across 40
   real test cases. Fixed with a forward-fill before thresholding.
2. **Unfiltered artifact spikes**: raw MAP values ranged from -70 to 346
   mmHg in a single real case (sensor/cautery-interference artifacts).
   Fixed by excluding physiologically implausible readings (outside
   20-180 mmHg) before thresholding.
3. **Per-window z-normalization erasing the actual predictive signal**
   (the most consequential bug): the first trained model scored AUC 0.53
   (chance level). Root-caused via a Mann-Whitney U test (p=5.7e-16)
   proving the real signal is the *absolute* pressure level (median MAP
   ~69 mmHg 5 min before a hypotensive onset vs. ~80 mmHg for matched
   negative windows) -- but per-window z-normalization forces every
   window to mean=0 regardless of its real absolute level, destroying
   exactly that signal. Fixed with a fixed clinical-anchor scale
   (`clip(20,250)` then `(x-100)/50`) instead of per-window normalization.
   This single fix took test AUC from 0.53 to 0.75.

## Results (real, held-out test split, case-level split to avoid leakage)

| Variant | Test AUC | Sens @ 0.5 | Spec @ 0.5 | Sens @ 85%-target threshold | Spec @ 85%-target threshold |
|---|---|---|---|---|---|
| `logs/` -- full CNN+LSTM ensemble | 0.752 | 56.0% | 80.5% | 86.5% | 44.0% |
| `logs_cnn_only/` -- CNN branch only | 0.751 | 62.4% | 75.4% | 86.5% | 42.7% |
| `logs_boosted_weight/` -- ensemble, 1.75x pos_weight | **0.753** | **77.7%** | **59.4%** | 86.0% | 45.6% |

**Finding**: all three variants converge to the same ~0.75 AUC ceiling.
Ablating the LSTM branch confirmed it adds negligible value (branch-alone
AUC 0.619 vs. CNN-alone 0.752, yet the learned ensemble weight favored
LSTM at 61%) -- a real architecture-simplification finding, not just a
negative result. The boosted-weight model is the most clinically usable
out of the box (77.7%/59.4% at the default threshold, vs. 56%/80.5% for
the original), without sacrificing ceiling performance.

Threshold selection was done properly throughout: swept on validation
data only, applied once to test, never touched during training.

## Status
Research-stage. No regulatory documentation package has been built for
this project yet (paused pending a decision on which model variant to
carry forward, and whether pushing past the AUC~0.75 ceiling via richer
multi-modal input -- e.g. adding ECG/PPG/capnography channels, as later
STEP-OP-lineage work does -- is worth pursuing before documenting).
