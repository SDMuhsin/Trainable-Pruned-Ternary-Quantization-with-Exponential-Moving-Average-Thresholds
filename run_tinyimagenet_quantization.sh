#!/bin/bash
# Run all TinyImageNet quantization experiments (after FP baseline is done)
# Usage: bash run_tinyimagenet_quantization.sh
# Prerequisite: FP baseline models must exist in ./results/TinyImageNet_RESNET50_FP_OW_2/model/

set -e

source env/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TORCH_HOME=./data
export HF_HOME=./data

# Verify FP models exist
FP_DIR="./results/TinyImageNet_RESNET50_FP_OW_2/model/"
if [ ! -d "$FP_DIR" ] || [ -z "$(ls -A $FP_DIR 2>/dev/null)" ]; then
    echo "ERROR: FP models not found in $FP_DIR"
    echo "Run the FP baseline first: python src/Experiments/train_model_base.py --parameters_file parameters_files/TinyImageNet/tinyimagenet_FP.json"
    exit 1
fi
echo "FP models found: $(ls $FP_DIR | wc -l) files"

echo "============================================"
echo "Starting TinyImageNet quantization pipeline"
echo "Start time: $(date)"
echo "============================================"

# Step 1: TTQ
echo ""
echo "[1/3] Running TTQ..."
echo "Start: $(date)"
python src/Experiments/experiment_TTQ.py --parameters_file parameters_files/TinyImageNet/tinyimagenet_TTQ.json
echo "TTQ done: $(date)"

# Step 2: pTTQ
echo ""
echo "[2/3] Running pTTQ..."
echo "Start: $(date)"
python src/Experiments/experiment_pTTQ.py --parameters_file parameters_files/TinyImageNet/tinyimagenet_pTTQ.json
echo "pTTQ done: $(date)"

# Step 3: EMA-pTTQ
echo ""
echo "[3/3] Running EMA-pTTQ..."
echo "Start: $(date)"
python src/Experiments/experiment_pTTQ_experimental.py --parameters_file parameters_files/TinyImageNet/tinyimagenet_experimental.json
echo "EMA-pTTQ done: $(date)"

echo ""
echo "============================================"
echo "All quantization experiments complete!"
echo "End time: $(date)"
echo "============================================"
