#!/usr/bin/env python3
"""Intraoperative hypotension prediction: one arterial-waveform window in, risk out.

DEVICE FRAMING. This emits a FORWARD-LOOKING RISK SCORE for a hypotensive event
(MAP < 65 mmHg, sustained) beginning ~5 minutes after the end of the input
window. It is not a diagnosis and not a measurement of current blood pressure --
the patient's present MAP is directly observable on the monitor and needs no
model. A low score does not rule out an impending event.

WHAT THE OPERATING POINT COSTS. The packaged threshold 0.2694 was selected on
the validation split alone, targeting 85% sensitivity, and applied once to the
held-out test split. It delivered sensitivity 0.865 at specificity 0.440 (test
AUC 0.752, n=5233). That specificity is the honest headline: at this operating
point roughly 56% of negative windows also alarm. The threshold is deliberately
set for a high-sensitivity monitoring role and is NOT appropriate for any use
where false alarms carry cost. Override it only with a fresh calibration.

SCALING IS A FIXED CLINICAL ANCHOR, AND THAT IS THE WHOLE MODEL. Training does:

    window = np.clip(window, 20.0, 250.0)
    window = (window - 100.0) / 50.0

Per-window z-normalisation was tried first and produced AUC 0.53 -- chance --
because it subtracts each window's OWN mean, erasing absolute pressure level.
That level IS the predictive signal: median MAP five minutes before an onset is
~69 mmHg versus ~80 mmHg for matched negatives (Mann-Whitney p=5.7e-16). So the
divisor here is a fixed constant, never computed from the input. Anything that
re-centres the window per-sample destroys the model exactly as thoroughly as it
did in training, and it will still return a confident-looking number.

CONSEQUENTLY THE INPUT MUST BE IN mmHg. A waveform arriving in kPa, in ADC
counts, or already normalised is not merely noisy -- it lands in a completely
different region of the fixed scale. Because the model has no way to detect
this, the loader checks the input's plausibility as a pressure trace and refuses
rather than scoring it. Same class of failure as feeding a MONOCHROME1 film to
an imaging model, and handled the same way: reject at the door, loudly.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/app")
from src.model import STEPOPEnsemble  # noqa: E402

EXPECTED_SAMPLES = 10_000     # 20 s @ 500 Hz, the only shape ever trained on
EXPECTED_RATE_HZ = 500
CLIP_LO, CLIP_HI = 20.0, 250.0
CENTER, SCALE = 100.0, 50.0

# A real arterial line sits well inside these. Outside them the trace is not a
# pressure waveform in mmHg, whatever else it may be.
PLAUSIBLE_MEDIAN = (30.0, 160.0)


def load_waveform(path: Path):
    """Return (float64 samples, provenance). Accepts .npy or single-column CSV."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float64).reshape(-1)
        src = "npy"
    else:
        arr = np.loadtxt(path, delimiter=",").astype(np.float64).reshape(-1)
        src = "csv"
    return arr, {"source": src, "n_samples_supplied": int(arr.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("waveform", help="20 s arterial waveform in mmHg: .npy or single-column .csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--weights", default="/app/weights/best_model.pt")
    ap.add_argument("--eval-json", default="/app/weights/test_eval_thresholded.json")
    ap.add_argument("--sample-rate", type=float, default=EXPECTED_RATE_HZ,
                    help="Hz of the supplied trace; must be 500, the rate trained on")
    ap.add_argument("--threshold", type=float, default=None,
                    help="overrides the packaged validation-calibrated threshold; rarely correct")
    ap.add_argument("--allow-implausible", action="store_true",
                    help="score even if the trace does not look like mmHg. Records a warning "
                         "in the output. Intended for phantom/bench testing only.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    p = Path(a.waveform)
    if not p.exists():
        sys.exit("no such file: %s" % p)

    ev = json.load(open(a.eval_json))
    thr = a.threshold if a.threshold is not None else float(ev["threshold"])

    raw, info = load_waveform(p)

    # --- shape. Resampling or padding to fit would silently change the input
    # distribution, so the mismatch is refused instead of repaired.
    if a.sample_rate != EXPECTED_RATE_HZ:
        sys.exit("sample rate %g Hz != %d Hz trained on; resampling is not performed "
                 "because it alters the waveform the model was fitted to"
                 % (a.sample_rate, EXPECTED_RATE_HZ))
    if raw.size != EXPECTED_SAMPLES:
        sys.exit("expected %d samples (20 s @ %d Hz), got %d. Supply exactly one 20 s window."
                 % (EXPECTED_SAMPLES, EXPECTED_RATE_HZ, raw.size))
    if not np.isfinite(raw).all():
        n_bad = int((~np.isfinite(raw)).sum())
        sys.exit("%d non-finite samples in the trace; the model cannot interpret them and "
                 "zero-filling would fabricate pressure readings" % n_bad)

    # --- units / plausibility, before any scaling
    med = float(np.median(raw))
    implausible = not (PLAUSIBLE_MEDIAN[0] <= med <= PLAUSIBLE_MEDIAN[1])
    if implausible and not a.allow_implausible:
        sys.exit("median sample %.3f is outside %.0f-%.0f mmHg, so this does not look like an "
                 "arterial pressure trace in mmHg. The model's scaling is a FIXED clinical "
                 "anchor -- a trace in other units lands in the wrong part of it and would "
                 "still produce a confident number. Convert to mmHg, or pass "
                 "--allow-implausible for bench testing." % (med, *PLAUSIBLE_MEDIAN))

    # --- artifact load. ~10-12% clipped is normal (disconnection/flush spikes);
    # a majority clipped means the line was not measuring the patient.
    clipped = int(((raw < CLIP_LO) | (raw > CLIP_HI)).sum())
    clip_frac = clipped / raw.size

    x = np.clip(raw, CLIP_LO, CLIP_HI)
    x = (x - CENTER) / SCALE                      # fixed anchor -- never per-window
    xt = torch.from_numpy(x.astype(np.float32)).view(1, 1, -1).to(dev)

    model = STEPOPEnsemble().to(dev)
    state = torch.load(a.weights, map_location="cpu")
    # strict=True: a partial load leaves an untrained branch in the ensemble and
    # still emits a full, plausible probability.
    model.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state,
                          strict=True)
    model.eval()

    with torch.no_grad():
        # fp32 -- autocast produced non-finite outputs on a large eval cohort in
        # a sibling project and silently destroyed a metric.
        logit = model(xt).float()
        prob = float(torch.sigmoid(logit).item())
        w_lstm = float(torch.sigmoid(model.raw_lstm_weight).item())

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device_output": "FORWARD-LOOKING RISK SCORE for a hypotensive event (MAP < 65 mmHg) "
                         "beginning ~5 min after this window. Not a diagnosis, not a current "
                         "blood-pressure measurement. A low score does not rule out an event.",
        "architecture": "STEP-OP-style ensemble: 1D CNN branch + BiLSTM branch, learned blend",
        "ensemble_lstm_weight": w_lstm,
        "input": {"path": str(p), **info, "sample_rate_hz": a.sample_rate,
                  "median_mmHg": med, "min_mmHg": float(raw.min()), "max_mmHg": float(raw.max()),
                  "clipped_samples": clipped, "clipped_fraction": round(clip_frac, 4)},
        "scaling": "clip to [%.0f, %.0f] mmHg then (x - %.0f)/%.0f -- FIXED anchor, not per-window"
                   % (CLIP_LO, CLIP_HI, CENTER, SCALE),
        "probability": prob,
        "threshold": thr,
        "threshold_source": ev.get("threshold_selection", "packaged test_eval_thresholded.json"),
        "flag": bool(prob >= thr),
        "operating_point": {"test_auc": ev.get("test_auc"),
                            "test_sensitivity": ev.get("test_sensitivity"),
                            "test_specificity": ev.get("test_specificity"),
                            "note": "specificity ~0.44: most negative windows also alarm at this "
                                    "high-sensitivity operating point"},
    }
    if implausible:
        result["WARNING_implausible_input"] = (
            "median %.3f mmHg is outside the plausible arterial range; scored only because "
            "--allow-implausible was passed. This output is not clinically meaningful." % med)
    if clip_frac > 0.5:
        result["WARNING_artifact"] = (
            "%.0f%% of samples fell outside [%.0f, %.0f] mmHg and were clipped; the line may "
            "have been disconnected or flushed rather than measuring the patient"
            % (clip_frac * 100, CLIP_LO, CLIP_HI))

    out_path = Path(a.out) if a.out else p.with_suffix(".hypotension_result.json")
    json.dump(result, open(out_path, "w"), indent=2)
    print("flag=%s  p=%.4f  thr=%.4f  median=%.1fmmHg  clipped=%.1f%%  device=%s"
          % (result["flag"], prob, thr, med, clip_frac * 100, dev))
    print("Wrote %s" % out_path)


if __name__ == "__main__":
    main()
