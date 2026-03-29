#!/bin/bash
# ============================================================================
# ImageNet-1K ConvNeXt FP Baseline — Nibi (SHARCNET/Waterloo)
# ============================================================================
#
# Trains the FP baseline model that all quantized methods will load.
# Must complete before running nibi_imagenet_quantized.sh.
#
# Nibi GPU nodes: 8x H100 80GB per node, Intel Xeon 6 Granite Rapids
# Request format: --gres=gpu:h100:N  (full GPUs, no MIG slices)
#
# Prerequisites:
#   - ImageNet-1K data at ./data/ImageNet/{train,val}/
#   - Run sbatch/download_cache.sh on login node first (if using HF datasets)
#
# Usage:
#   ./sbatch/nibi_imagenet_fp.sh --account def-myprof
#   ./sbatch/nibi_imagenet_fp.sh --account def-myprof --gpus 4
#   ./sbatch/nibi_imagenet_fp.sh --account def-myprof --gpus 8 --epochs 90
#
# ============================================================================

ACCOUNT=""
NUM_GPUS=4
EPOCHS=50
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
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --account SLURM_ACCOUNT [--gpus NUM_GPUS] [--epochs N]"
            exit 1
            ;;
    esac
done

if [[ -z "$ACCOUNT" ]]; then
    echo "ERROR: --account is required on Nibi"
    echo "Usage: $0 --account def-myprof [--gpus $NUM_GPUS] [--epochs $EPOCHS]"
    exit 1
fi

# Scale resources per GPU: 24 cores + 30GB RAM per H100 on a Nibi GPU node
CPUS=$((NUM_GPUS * 12))
MEM=$((NUM_GPUS * 30000))

mkdir -p ./logs ./results

echo "============================================"
echo "ImageNet-1K ConvNeXt — FP Baseline (Nibi)"
echo "  GPUs: ${NUM_GPUS}x H100 80GB"
echo "  Epochs: ${EPOCHS}"
echo "  Time limit: 2-00:00:00"
echo "============================================"

sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=imgnet_fp
#SBATCH --output=./logs/imgnet_fp_%j.out
#SBATCH --error=./logs/imgnet_fp_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100:${NUM_GPUS}
#SBATCH --mem=${MEM}M
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --account=${ACCOUNT}

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

export HF_HOME=\$(pwd)/data
export TORCH_HOME=\$(pwd)/data
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=\$PYTHONPATH:\$(pwd)
mkdir -p \$HF_HOME

echo '========================================'
echo "Job: ImageNet-1K ConvNeXt FP Baseline"
echo "Cluster: Nibi"
echo "GPUs: ${NUM_GPUS}x H100 80GB (DataParallel)"
echo "Epochs: ${EPOCHS}, Batch: 256"
echo "Started: \$(date)"
echo '========================================'
nvidia-smi

# Write a temporary config with the requested epoch count
python3 -c "
import json
with open('./parameters_files/ImageNet/imagenet_convnext_fp_prod.json') as f:
    cfg = json.load(f)
cfg['nb_epochs'] = ${EPOCHS}
with open('/tmp/nibi_fp_\${SLURM_JOB_ID}.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Config written with nb_epochs=${EPOCHS}')
"

python3 -u src/Experiments/train_model_base.py \
    --parameters_file=/tmp/nibi_fp_\${SLURM_JOB_ID}.json

echo '========================================'
echo "Finished: \$(date)"
echo '========================================'
EOF
)

echo "  [$sbatch_id] imgnet_fp  (${NUM_GPUS}x H100, ${EPOCHS}ep, 2d)"
echo ""
echo "============================================"
echo "FP baseline submitted. Wait for completion, then run:"
echo "  ./sbatch/nibi_imagenet_quantized.sh --account $ACCOUNT"
echo "FP model saved to:"
echo "  ./results/PROD_ImageNet_CONVNEXT_FP_OW_0/model/"
echo "============================================"
