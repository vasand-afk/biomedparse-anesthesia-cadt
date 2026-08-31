# Deployment container

**Image:** `ghcr.io/vasand-afk/biomedparse-anesthesia-cadt:validation-20260831`
**Base:** `ghcr.io/vasand-afk/biomedparse-base@sha256:0716b93d…f289` (digest-pinned)
**Trained under:** torch 2.8.0+cu128 (`run_config.json`; `environment.json` agrees)
**Runs under:** torch 2.6.0+cu124 (shared base) — equivalence check NOT yet run, see below

## Operating point

Threshold **0.2694**, selected on the validation split alone targeting 85%
sensitivity and applied once to the held-out test split. Test: AUC 0.752,
sensitivity 0.865, **specificity 0.440**, n=5233.

The specificity is the honest headline. At this operating point roughly 56% of
negative windows also alarm. That is a deliberate high-sensitivity monitoring
choice, not a defect, but it makes the device unsuitable for any workflow where
a false alarm carries cost.

## What the container enforces

The scaling is a **fixed clinical anchor** — `clip(20, 250)` then `(x-100)/50`
— never computed from the input. Per-window z-normalisation was tried during
development and produced AUC 0.53 (chance), because it erases absolute pressure
level, which is the actual predictive signal (median MAP 5 min pre-onset ~69
mmHg vs ~80 for matched negatives, Mann-Whitney p=5.7e-16).

Because the anchor is fixed, a units error does not degrade into noise — it
relocates the entire trace to a different part of the scale and still returns a
confident number. The loader therefore **refuses** rather than scores:

| condition | behaviour |
|---|---|
| median outside 30–160 mmHg | refuse (units error / not a pressure trace) |
| sample count ≠ 10,000 | refuse — never pad or resample |
| any non-finite sample | refuse — never zero-fill |
| sample rate ≠ 500 Hz | refuse — never resample |
| >50% samples clipped | score, with `WARNING_artifact` |

The build itself asserts the weights load `strict=True` and that the packaged
threshold is still 0.2694, so a swapped or missing artifact fails the build
rather than surfacing as a runtime `KeyError` — or worse, a silent default of 0.5.

## Synthetic verification (2026-08-31)

Phantom waveforms, plumbing only — the probabilities are meaningless as
performance:

```
normal_80.npy         p=0.4125  flag=True   median 83.1 mmHg
low_65.npy            p=0.6152  flag=True   median 68.1
high_100.npy          p=0.1899  flag=False  median 103.1
units_normalised.npy  REFUSED (median 0.616)
units_kpa.npy         REFUSED (median 11.06)
short_10s.npy         REFUSED (5000 samples)
has_nan.npy           REFUSED (20 non-finite)
disconnected.npy      REFUSED (median 900)
normal_80.csv         p=0.4125  — identical to the .npy path
```

Risk falls monotonically as MAP rises (0.615 → 0.413 → 0.190), the
physiologically expected direction.

**Known gap in this test.** `disconnected.npy` was intended to exercise the
`>50% clipped` artifact warning, but its median (900) tripped the earlier
plausibility gate, so it was refused before reaching that branch. Refusal is
the correct behaviour for that input, but it means **the `WARNING_artifact`
path is currently untested**. Constructing a case that is majority-clipped
while keeping a plausible median would be needed to cover it.

## Not yet done

- **Runtime equivalence check** (2.8.0 vs the deployed 2.6.0), as run for
  derm/ultrasound/amd/glaucoma/kneeoa. This model is 1-D signal rather than
  imaging, so the sibling probe does not apply unmodified.
- **No risk analysis, labeling, or predicate comparison exists for this
  project.** Containerisation packages a model; it does not validate one. This
  document describes an artifact, not a cleared device.
