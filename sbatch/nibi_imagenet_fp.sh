#!/bin/bash
# ============================================================================
# ImageNet-1K ConvNeXt FP Baseline — Nibi (SHARCNET/Waterloo)
# ============================================================================
#
# Trains the FP baseline model that all quantized methods will load.
# Must complete before running nibi_imagenet_quantized.sh.
#
# Data strategy: ImageNet stored as tar files in /project to avoid file count
# quota. Extracted to $SLURM_TMPDIR (node-local NVMe) at job start.
#
# Nibi GPU nodes: 8x H100 80GB per node, Intel Xeon 6 Granite Rapids
# Request format: --gres=gpu:h100:N  (full GPUs, no MIG slices)
#
# Prerequisites:
#   - data/ImageNet_train.tar and data/ImageNet_val.tar in project dir
#
# Usage:
#   ./sbatch/nibi_imagenet_fp.sh --account def-myprof
#   ./sbatch/nibi_imagenet_fp.sh --account def-myprof --gpus 4 --epochs 50
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

for f in ./data/ImageNet_train.tar ./data/ImageNet_val.tar; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f not found."
        exit 1
    fi
done

CPUS=$((NUM_GPUS * 12))
MEM=$((NUM_GPUS * 30000))

mkdir -p ./logs ./results

echo "============================================"
echo "ImageNet-1K ConvNeXt — FP Baseline (Nibi)"
echo "  GPUs: ${NUM_GPUS}x H100 80GB"
echo "  Epochs: ${EPOCHS}, Time: 2d"
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

module load gcc arrow scipy-stack cuda
source ./env/bin/activate

export HF_HOME=\$(pwd)/data
export TORCH_HOME=\$(pwd)/data
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=\$PYTHONPATH:\$(pwd)

echo '========================================'
echo "Job: ImageNet-1K ConvNeXt FP Baseline"
echo "Cluster: Nibi — ${NUM_GPUS}x H100 (DataParallel)"
echo "Epochs: ${EPOCHS}, Batch: 256"
echo "Started: \$(date)"
echo '========================================'
nvidia-smi

# Extract ImageNet from tar to node-local NVMe
echo "Extracting ImageNet to \$SLURM_TMPDIR..."
mkdir -p \$SLURM_TMPDIR/ImageNet
tar xf ./data/ImageNet_train.tar -C \$SLURM_TMPDIR/ImageNet &
tar xf ./data/ImageNet_val.tar -C \$SLURM_TMPDIR/ImageNet &
wait
echo "Extraction complete: \$(ls \$SLURM_TMPDIR/ImageNet/train | wc -l) train classes, \$(ls \$SLURM_TMPDIR/ImageNet/val | wc -l) val classes"

# Write config pointing at SLURM_TMPDIR
python3 -c "
import json, os
with open('./parameters_files/ImageNet/imagenet_convnext_fp_prod.json') as f:
    cfg = json.load(f)
cfg['nb_epochs'] = ${EPOCHS}
cfg['dataset_folder'] = os.environ['SLURM_TMPDIR'] + '/ImageNet/'
with open('/tmp/nibi_fp_\${SLURM_JOB_ID}.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Config: nb_epochs=${EPOCHS}, dataset_folder:', cfg['dataset_folder'])
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
echo "FP model saved to: ./results/PROD_ImageNet_CONVNEXT_FP_OW_0/model/"
echo "Then run: ./sbatch/nibi_imagenet_quantized.sh --account $ACCOUNT"
