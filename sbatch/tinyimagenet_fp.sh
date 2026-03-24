#!/bin/bash
# ============================================================================
# TinyImageNet ConvNeXt FP Baseline — SLURM Submission Script
# ============================================================================
#
# Trains the FP baseline model (50 epochs, 1 rep) that all quantized methods
# will load. Must complete before running tinyimagenet_quantized.sh.
#
# Usage:
#   ./sbatch/tinyimagenet_fp.sh
#   ./sbatch/tinyimagenet_fp.sh --account def-myprof
#
# ============================================================================

ACCOUNT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT]"
            exit 1
            ;;
    esac
done

account_line=""
if [[ -n "$ACCOUNT" ]]; then
    account_line="#SBATCH --account=$ACCOUNT"
fi

mkdir -p ./logs ./results

echo "============================================"
echo "TinyImageNet ConvNeXt — FP Baseline"
echo "============================================"

sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=tinyimg_fp
#SBATCH --output=./logs/tinyimg_fp_%j.out
#SBATCH --error=./logs/tinyimg_fp_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --mem=32000M
#SBATCH --cpus-per-task=4
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
echo "Job: TinyImageNet ConvNeXt FP Baseline"
echo "Epochs: 50, Reps: 1"
echo "GPU: h100_3g.40gb"
echo "Started: \$(date)"
echo '========================================'
nvidia-smi
export PYTHONPATH=\$PYTHONPATH:\$(pwd)

python3 src/Experiments/train_model_base.py \
    --parameters_file=./parameters_files/TinyImageNet/tinyimagenet_convnext_fp_prod.json

echo '========================================'
echo "Finished: \$(date)"
echo '========================================'
EOF
)

echo "  [$sbatch_id] tinyimg_fp  (ConvNeXt FP baseline, 50ep, 2d, h100_3g.40gb)"
echo ""
echo "============================================"
echo "FP baseline submitted. Wait for it to complete before running:"
echo "  ./sbatch/tinyimagenet_quantized.sh"
echo "FP model will be saved to:"
echo "  ./results/PROD_TinyImageNet_CONVNEXT_FP_OW_0/model/"
echo "============================================"
