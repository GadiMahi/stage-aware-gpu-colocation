#!/usr/bin/env bash
# Full pipeline, start to finish. Roughly one minute on the simulated backend;
# roughly seven GPU-hours on a real 2xT4 node.
set -euo pipefail

echo "=== environment ==="
python -m sagc probe

echo; echo "=== tests ==="
python -m pytest tests/ -q

echo; echo "=== M2 profile zoo ==="
python -m sagc profile --reps "${REPS:-3}"

echo; echo "=== M3 pairwise sweep ==="
python -m sagc sweep --reps "${REPS:-3}" --thread-pcts "${THREAD_PCTS:-100,50}"

echo; echo "=== M4 predictor and ablations ==="
python -m sagc train

echo; echo "=== M5/M6 scheduler, simulator, ILP gap ==="
python -m sagc schedule --pipelines "${PIPELINES:-20}"

echo; echo "=== figures ==="
python -m sagc figures

echo; echo "artefacts:  data/  results/  figures/"
