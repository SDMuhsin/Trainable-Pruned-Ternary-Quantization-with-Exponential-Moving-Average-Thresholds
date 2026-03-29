#!/bin/bash
# ============================================================================
# ImageNet-1K ConvNeXt FP Baseline — SLURM Submission Script
# ============================================================================
#
# Trains the FP baseline model (90 epochs, 1 rep) that all quantized methods
# will load. Must complete before running imagenet_quantized.sh.
#
# Prerequisites:
#   - ImageNet-1K data at ./data/ImageNet/{train,val}/ (rsync from local)
#
# Usage:
#   ./sbatch/imagenet_fp.sh
#   ./sbatch/imagenet_fp.sh --account def-myprof
#   ./sbatch/imagenet_fp.sh --account def-myprof --gpus 4
#
# ============================================================================

ACCOUNT=""
NUM_GPUS=4
while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT] [--gpus NUM_GPUS]"
            exit 1
            ;;
    esac
done

account_line=""
if [[ -n "$ACCOUNT" ]]; then
    account_line="#SBATCH --account=$ACCOUNT"
fi

CPUS=$((NUM_GPUS * 8))
MEM=$((NUM_GPUS * 32000))

mkdir -p ./logs ./results

echo "============================================"
echo "ImageNet-1K ConvNeXt — FP Baseline"
echo "============================================"

sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=imgnet_fp
#SBATCH --output=./logs/imgnet_fp_%j.out
#SBATCH --error=./logs/imgnet_fp_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --gpus=h100:${NUM_GPUS}
#SBATCH --mem=${MEM}M
#SBATCH --cpus-per-task=${CPUS}
$account_line

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

export HF_HOME=\$(pwd)/data
export TORCH_HOME=\$(pwd)/data
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
mkdir -p \$HF_HOME

echo '========================================'
echo "Job: ImageNet-1K ConvNeXt FP Baseline"
echo "Epochs: 90, Reps: 1, Batch: 256"
echo "GPUs: ${NUM_GPUS}x h100 (DataParallel)"
echo "Started: \$(date)"
echo '========================================'
nvidia-smi
export PYTHONPATH=\$PYTHONPATH:\$(pwd)

python3 src/Experiments/train_model_base.py \
    --parameters_file=./parameters_files/ImageNet/imagenet_convnext_fp_prod.json

echo '========================================'
echo "Finished: \$(date)"
echo '========================================'
EOF
)

echo "  [$sbatch_id] imgnet_fp  (ConvNeXt FP baseline, 90ep, 7d, ${NUM_GPUS}x h100)"
echo ""
echo "============================================"
echo "FP baseline submitted. Wait for it to complete before running:"
echo "  ./sbatch/imagenet_quantized.sh"
echo "FP model will be saved to:"
echo "  ./results/PROD_ImageNet_CONVNEXT_FP_OW_0/model/"
echo "============================================"
