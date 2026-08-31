# Pinned by DIGEST, not by tag. `biomedparse-base:cu124` was a mutable local
# tag that existed only in one host's daemon; a rebuild of that tag would have
# silently changed every device built on it. The digest below is immutable.
# Human-readable equivalent: ghcr.io/vasand-afk/biomedparse-base:cu124-20260831
FROM ghcr.io/vasand-afk/biomedparse-base@sha256:0716b93d4c89e8ac90e13d3a24e03d631bd4fee8071b7171ed71d02f4fe5f289

LABEL org.opencontainers.image.description="Intraoperative hypotension risk from a 20 s arterial waveform (1D CNN + BiLSTM ensemble)"
LABEL org.opencontainers.image.source="https://github.com/vasand-afk/biomedparse-anesthesia-cadt"

# This model is 1-D signal, not imaging: it needs torch and numpy from the base
# and nothing else. torchvision/pydicom come along in the shared base and are
# simply unused here -- kept rather than stripped so this image shares the
# base's layers byte-for-byte, which is the point of having a shared base.

WORKDIR /app
COPY src/ /app/src/
COPY deployment/predict.py /app/predict.py

# Weights and the calibrated operating point travel WITH the image. A container
# that fetches its threshold at runtime can silently score against a different
# operating point than the one that was validated.
COPY logs/run_20260829T112319Z/best_model.pt /app/weights/best_model.pt
COPY logs/run_20260829T112319Z/test_eval_thresholded.json /app/weights/test_eval_thresholded.json

# Fail the BUILD if the packaged artifacts are not the ones that were validated.
# A missing or mismatched threshold file would otherwise surface as a runtime
# KeyError on a clinical machine, or worse, as a silent default of 0.5.
RUN python3 -c "\
import json, torch, sys; \
sys.path.insert(0, '/app'); \
from src.model import STEPOPEnsemble; \
ev = json.load(open('/app/weights/test_eval_thresholded.json')); \
assert abs(ev['threshold'] - 0.2694) < 1e-9, 'threshold changed: %r' % ev['threshold']; \
m = STEPOPEnsemble(); \
st = torch.load('/app/weights/best_model.pt', map_location='cpu'); \
m.load_state_dict(st.get('state_dict', st) if isinstance(st, dict) else st, strict=True); \
m.eval(); \
x = torch.zeros(1, 1, 10000); \
p = float(torch.sigmoid(m(x)).item()); \
assert 0.0 < p < 1.0, 'degenerate output %r' % p; \
print('BUILD CHECK OK: weights load strict=True, threshold', ev['threshold'], 'smoke p', round(p, 6))"

ENTRYPOINT ["python3", "/app/predict.py"]
CMD ["--help"]
