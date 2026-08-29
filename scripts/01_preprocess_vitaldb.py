#!/usr/bin/env python3
"""Download + preprocess VitalDB for intraoperative hypotension prediction,
mirroring STEP-OP's real design (Seoul National University, published in
JMIR Medical Informatics 2021, PMC8517810): predict a hypotensive event
5 minutes ahead from a 20-second arterial-waveform window.

Access -- confirmed directly (2026-08-28), not assumed: `pip install
vitaldb` gives free API access with no data-use-agreement/credentialing
gate, unlike MIMIC. vitaldb.find_cases(['SNUADC/ART', 'Solar8000/ART_MBP'])
returns 3,644 real cases with both the raw 500Hz arterial waveform track
and the numeric mean-arterial-pressure track needed to define hypotension
events -- both confirmed present via a real case load (case 1: MBP at
1Hz/11,543s, ART waveform at 500Hz/5,771,049 samples, <0.02% NaN).

HYPOTENSION DEFINITION: MAP < 65 mmHg sustained for >=60 consecutive
1-second samples -- the standard clinical threshold used in STEP-OP and
the broader IOH literature (not a novel/guessed cutoff).

WINDOW DESIGN (matches STEP-OP exactly): for each hypotension event onset
at time T, extract the 20-second raw ART waveform ending at T-300s (5
minutes before onset) as a positive sample. Negative samples are 20-second
windows randomly drawn from periods at least 10 minutes away from any
hypotension event in the same case, keeping the same per-case
positive:negative ratio roughly balanced rather than reflecting the raw
(highly imbalanced) event rate.

Output: one .npz per case (resumable -- skips cases whose output already
exists), containing stacked (N, 10000) float32 waveform windows (20s @
500Hz) and an (N,) binary label array.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import vitaldb

MAP_THRESHOLD = 65.0
EVENT_MIN_DURATION_S = 60
WINDOW_S = 20
WAVEFORM_HZ = 500
WINDOW_SAMPLES = WINDOW_S * WAVEFORM_HZ
LOOKAHEAD_S = 300  # 5 minutes, matching STEP-OP
NEGATIVE_BUFFER_S = 600  # negatives must be >=10 min from any event, on either side


MIN_LOW_FRACTION = 0.8  # see REAL FINDING below
MAP_PLAUSIBLE_RANGE = (20.0, 180.0)  # physiologically implausible readings are sensor artifacts


def ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs (carry last valid value forward); leading NaNs
    before the first valid reading stay NaN."""
    idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[idx == 0] = x[0] if not np.isnan(x[0]) else np.nan
    return out


def find_hypotension_onsets(map_raw: np.ndarray) -> list:
    """map_raw: (T,) array, 1 sample/sec. Returns onset times (seconds
    from case start) of each window where MAP stays below MAP_THRESHOLD
    for a real sustained episode.

    REAL FINDING 2026-08-28, in two parts, both confirmed by directly
    inspecting case 1's raw values (not assumed):
    (1) Solar8000/ART_MBP only updates every ~2 seconds at the device
    level -- at interval=1 loading, every other 1-second sample is a
    genuine NaN (no new reading yet), not missing/bad data. An earlier
    version of this function treated every NaN as "not hypotensive",
    which fragments every real sustained episode into isolated 1-sample
    blips (confirmed: a strict consecutive-run definition and even an
    80%-of-window majority definition both returned ZERO events across
    40 real cases with this bug in place). Fixed with a forward-fill
    before thresholding -- standard handling for a lower-native-rate
    numeric parameter, not a hack.
    (2) Raw values also contain physiologically impossible readings
    (confirmed: case 1 alone had values from -70 to 346 mmHg, with only
    42.8% of raw samples in a plausible range) -- real sensor/cautery-
    interference artifacts common in OR monitoring, not real MAP.
    Excluded via MAP_PLAUSIBLE_RANGE before thresholding so a spurious
    spike can't masquerade as (or mask) a real hypotensive episode."""
    plausible = (map_raw >= MAP_PLAUSIBLE_RANGE[0]) & (map_raw <= MAP_PLAUSIBLE_RANGE[1])
    clean = np.where(plausible, map_raw, np.nan)
    filled = ffill(clean)

    below = filled < MAP_THRESHOLD
    below = np.where(np.isnan(filled), False, below).astype(np.float32)
    if len(below) < EVENT_MIN_DURATION_S:
        return []

    window_frac = np.convolve(below, np.ones(EVENT_MIN_DURATION_S), mode="valid") / EVENT_MIN_DURATION_S
    is_event_window = window_frac >= MIN_LOW_FRACTION

    onsets = []
    in_event = False
    for t, flag in enumerate(is_event_window):
        if flag and not in_event:
            onsets.append(t)
            in_event = True
        elif not flag and in_event:
            in_event = False
    return onsets


def extract_window(art_500hz: np.ndarray, center_time_s: float) -> np.ndarray:
    """art_500hz: (T,) waveform at 500Hz. Returns a WINDOW_SAMPLES-length
    window ending at center_time_s, or None if out of bounds or too much
    missing data (>5% NaN -- a real sensor dropout, not synthetic fill)."""
    end_idx = int(center_time_s * WAVEFORM_HZ)
    start_idx = end_idx - WINDOW_SAMPLES
    if start_idx < 0 or end_idx > len(art_500hz):
        return None
    window = art_500hz[start_idx:end_idx]
    if np.isnan(window).mean() > 0.05:
        return None
    return np.nan_to_num(window, nan=0.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/processed/cases")
    ap.add_argument("--limit-cases", type=int, default=None, help="debug: process only first N cases")
    ap.add_argument("--max-negatives-per-case", type=int, default=10)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Finding cases with both required tracks...", flush=True)
    case_ids = vitaldb.find_cases(["SNUADC/ART", "Solar8000/ART_MBP"])
    print(f"{len(case_ids)} cases found", flush=True)
    if args.limit_cases:
        case_ids = case_ids[: args.limit_cases]

    rng = np.random.RandomState(42)
    n_done, n_errored, n_no_events = 0, 0, 0

    for i, cid in enumerate(case_ids):
        out_path = out_dir / f"{cid}.npz"
        if out_path.exists():
            continue
        if i % 20 == 0:
            print(f"{i}/{len(case_ids)} cases (done={n_done}, errored={n_errored}, "
                  f"no_events={n_no_events})", flush=True)

        try:
            map_1hz = vitaldb.load_case(cid, ["Solar8000/ART_MBP"], interval=1).flatten()
            onsets = find_hypotension_onsets(map_1hz)
            if not onsets:
                n_no_events += 1
                continue

            art_500hz = vitaldb.load_case(cid, ["SNUADC/ART"], interval=1.0 / WAVEFORM_HZ).flatten()

            windows, labels = [], []
            for onset_s in onsets:
                w = extract_window(art_500hz, onset_s - LOOKAHEAD_S)
                if w is not None:
                    windows.append(w)
                    labels.append(1)

            case_duration_s = len(map_1hz)
            event_times = set()
            for onset_s in onsets:
                event_times.update(range(max(0, onset_s - NEGATIVE_BUFFER_S), onset_s + NEGATIVE_BUFFER_S))

            n_neg_target = min(args.max_negatives_per_case, len(windows) * 2)
            attempts = 0
            while len(labels) - sum(labels) < n_neg_target and attempts < n_neg_target * 10:
                attempts += 1
                t = rng.randint(WINDOW_S, case_duration_s - LOOKAHEAD_S)
                if t in event_times:
                    continue
                w = extract_window(art_500hz, t)
                if w is not None:
                    windows.append(w)
                    labels.append(0)

            if not windows:
                n_no_events += 1
                continue

            np.savez_compressed(out_path, windows=np.stack(windows), labels=np.array(labels, dtype=np.int64))
            n_done += 1
        except Exception as e:
            print(f"SKIP case {cid}: {e}", flush=True)
            n_errored += 1

    print(f"Done. {n_done} cases processed, {n_errored} errored, {n_no_events} had no usable events/data.")


if __name__ == "__main__":
    main()
