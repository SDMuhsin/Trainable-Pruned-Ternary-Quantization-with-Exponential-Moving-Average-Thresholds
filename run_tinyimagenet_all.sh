#!/bin/bash
# Run all TinyImageNet experiments sequentially: FP baseline, then TTQ, pTTQ, EMA-pTTQ
# Usage: bash run_tinyimagenet_all.sh
# Expected total time: ~35-44 hours (4 methods × ~8.8 hours each)

set -e

source env/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TORCH_HOME=./data
export HF_HOME=./data

echo "============================================"
echo "Starting TinyImageNet experiment pipeline"
echo "Start time: $(date)"
echo "============================================"

# Step 1: FP Baseline
echo ""
echo "[1/4] Running FP Baseline..."
echo "Start: $(date)"
python src/Experiments/train_model_base.py --parameters_file parameters_files/TinyImageNet/tinyimagenet_FP.json
echo "FP Baseline done: $(date)"

# Step 2: TTQ
echo ""
echo "[2/4] Running TTQ..."
echo "Start: $(date)"
python src/Experiments/experiment_TTQ.py --parameters_file parameters_files/TinyImageNet/tinyimagenet_TTQ.json
echo "TTQ done: $(date)"

# Step 3: pTTQ
echo ""
echo "[3/4] Running pTTQ..."
echo "Start: $(date)"
python src/Experiments/experiment_pTTQ.py --parameters_file parameters_files/TinyImageNet/tinyimagenet_pTTQ.json
echo "pTTQ done: $(date)"

# Step 4: EMA-pTTQ
echo ""
echo "[4/4] Running EMA-pTTQ..."
echo "Start: $(date)"
python src/Experiments/experiment_pTTQ_experimental.py --parameters_file parameters_files/TinyImageNet/tinyimagenet_experimental.json
echo "EMA-pTTQ done: $(date)"

echo ""
echo "============================================"
echo "All TinyImageNet experiments complete!"
echo "End time: $(date)"
echo "============================================"
