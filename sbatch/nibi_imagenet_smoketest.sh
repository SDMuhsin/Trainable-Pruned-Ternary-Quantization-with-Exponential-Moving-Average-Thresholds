#!/bin/bash
# ============================================================================
# ImageNet-1K ConvNeXt FP Smoke Test — Nibi
# ============================================================================
#
# Quick sanity check: 1 epoch on 1% of data, single GPU, 20 min limit.
# Verifies data loading, model creation, forward/backward, and model saving.
#
# Data strategy: ImageNet stored as tar files in /project to avoid file count
# quota. Extracted to $SLURM_TMPDIR (node-local NVMe) at job start.
#
# Prerequisites:
#   - data/ImageNet_train.tar and data/ImageNet_val.tar in project dir
#
# Usage:
#   ./sbatch/nibi_imagenet_smoketest.sh --account def-myprof
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
            echo "Usage: $0 --account SLURM_ACCOUNT"
            exit 1
            ;;
    esac
done

if [[ -z "$ACCOUNT" ]]; then
    echo "ERROR: --account is required"
    exit 1
fi

# Verify tar files exist
for f in ./data/ImageNet_train.tar ./data/ImageNet_val.tar; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f not found. Create with:"
        echo "  cd data && tar cf ImageNet_train.tar -C ImageNet train"
        echo "  cd data && tar cf ImageNet_val.tar -C ImageNet val"
        exit 1
    fi
done

mkdir -p ./logs ./results

echo "============================================"
echo "ImageNet-1K Smoke Test (Nibi)"
echo "  1x H100, 1 epoch, 1% data, 20 min"
echo "============================================"

sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=imgnet_smoke
#SBATCH --output=./logs/imgnet_smoke_%j.out
#SBATCH --error=./logs/imgnet_smoke_%j.err
#SBATCH --time=0-00:20:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=30000M
#SBATCH --cpus-per-task=12
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
echo "Smoke Test: ImageNet-1K ConvNeXt FP"
echo "Cluster: Nibi — 1x H100"
echo "1 epoch, 1% data, batch 64"
echo "Started: \$(date)"
echo '========================================'
nvidia-smi

# Extract ImageNet from tar to node-local NVMe (no file quota, fast I/O)
echo "Extracting ImageNet to \$SLURM_TMPDIR..."
mkdir -p \$SLURM_TMPDIR/ImageNet
tar xf ./data/ImageNet_train.tar -C \$SLURM_TMPDIR/ImageNet &
tar xf ./data/ImageNet_val.tar -C \$SLURM_TMPDIR/ImageNet &
wait
echo "Extraction complete: \$(ls \$SLURM_TMPDIR/ImageNet/train | wc -l) train classes, \$(ls \$SLURM_TMPDIR/ImageNet/val | wc -l) val classes"

python3 -c "
import json
cfg = {
    'exp_id': 'SMOKE_NIBI_ImageNet_CONVNEXT_FP',
    'lr': 0.001, 'nb_repetitions': 1,
    'batch_size_train': 64, 'batch_size_test': 64,
    'weight_decay': 0, 'nb_epochs': 1, 'loss_function': 'CE',
    'model_type': '2DCNN', 'model_to_use': 'imagenetconvnext',
    'do_normalization_weights': False, 'device': 'cuda',
    'use_soft_labels': False, 'balance_dataset': False,
    'compute_class_weights': False, 'separate_val_ds': False,
    'dataset_type': 'ImageNet', 'percentage_samples_keep': 0.01,
    'dataset_folder': '', 'num_workers': 8
}
import os
cfg['dataset_folder'] = os.environ['SLURM_TMPDIR'] + '/ImageNet/'
with open('/tmp/nibi_smoke_\${SLURM_JOB_ID}.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Config written, dataset_folder:', cfg['dataset_folder'])
"

python3 -u src/Experiments/train_model_base.py \
    --parameters_file=/tmp/nibi_smoke_\${SLURM_JOB_ID}.json

echo '========================================'
echo "Finished: \$(date)"
echo '========================================'
EOF
)

echo "  [$sbatch_id] imgnet_smoke  (1x H100, 1ep, 1% data, 20min)"
