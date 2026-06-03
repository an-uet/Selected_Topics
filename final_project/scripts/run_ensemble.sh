#!/bin/bash
# Model Ensemble + TTA inference → submission CSV
# Run from project root: bash scripts/run_ensemble.sh

set -e
cd "$(dirname "$0")/.."

GPU=${1:-5}   # default GPU 0, override: bash scripts/run_ensemble.sh 1

echo "=== Step 1: Ensemble inference (GPU $GPU, TTA enabled) ==="
PYTHONPATH=$(pwd) python scripts/ensemble_infer.py \
    --lq-dir  /mnt/HDD4/anlt/data/test/lr \
    --out-dir results/MSW_ensemble/visualization/Single \
    --gpu     "$GPU"

echo ""
echo "=== Step 2: Generate submission CSV ==="
python gen.py \
    -f results/MSW_ensemble/visualization/Single \
    -s results/MSW_ensemble.csv

echo ""
echo "Done! Submission file: results/MSW_ensemble.csv"
