#!/bin/bash
# ============================================================================
# ImageNet-1K ConvNeXt Quantized Methods — Nibi (SHARCNET/Waterloo)
# ============================================================================
#
# Submits all quantized methods in parallel:
#   - DoReFaNet (1 job)
#   - TTQ (1 job)
#   - pTTQ (3 jobs: k=0.5, k=1.0, k=1.2)
#   - EMA-pTTQ (3 jobs: k=0.5, k=1.0, k=1.2)
#
# Data strategy: ImageNet stored as tar files in /project to avoid file count
# quota. Each job extracts to its own $SLURM_TMPDIR (node-local NVMe).
#
# PREREQUISITE: FP baseline must be trained first!
#   ./sbatch/nibi_imagenet_fp.sh --account def-myprof
#   (wait for completion)
#   ./sbatch/nibi_imagenet_quantized.sh --account def-myprof
#
# Usage:
#   ./sbatch/nibi_imagenet_quantized.sh --account def-myprof
#   ./sbatch/nibi_imagenet_quantized.sh --account def-myprof --gpus 4 --epochs 50
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
    exit 1
fi

for f in ./data/ImageNet_train.tar ./data/ImageNet_val.tar; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f not found."
        exit 1
    fi
done

FP_MODEL_DIR="./results/PROD_ImageNet_CONVNEXT_FP_OW_0/model/"
if [[ ! -d "$FP_MODEL_DIR" ]]; then
    echo "ERROR: FP model not found: $FP_MODEL_DIR"
    echo "Run ./sbatch/nibi_imagenet_fp.sh first."
    exit 1
fi

CPUS=$((NUM_GPUS * 12))
MEM=$((NUM_GPUS * 30000))
job_count=0

echo "============================================"
echo "ImageNet-1K ConvNeXt — Quantized (Nibi)"
echo "  ${NUM_GPUS}x H100, ${EPOCHS}ep, 2d limit"
echo "============================================"

# ============================================================================
# HELPER: Submit a single job
# ============================================================================
submit_job() {
    local job_name="$1"
    local python_cmd="$2"
    local desc="$3"
    local config_file="$4"

    local sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=./logs/${job_name}_%j.out
#SBATCH --error=./logs/${job_name}_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:h100:${NUM_GPUS}
#SBATCH --mem=${MEM}M
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --account=${ACCOUNT}

module load gcc arrow scipy-stack cuda
source ./env/bin/activate

export PYTHONNOUSERSITE=1
export HF_HOME=\$(pwd)/data
export TORCH_HOME=\$(pwd)/data
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=\$PYTHONPATH:\$(pwd)

echo '========================================'
echo "Job: $job_name — $desc"
echo "Cluster: Nibi — ${NUM_GPUS}x H100 (DataParallel)"
echo "Epochs: ${EPOCHS}"
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
with open('$config_file') as f:
    cfg = json.load(f)
cfg['nb_epochs'] = ${EPOCHS}
cfg['dataset_folder'] = os.environ['SLURM_TMPDIR'] + '/ImageNet/'
with open('/tmp/nibi_${job_name}_\${SLURM_JOB_ID}.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Config: nb_epochs=${EPOCHS}, dataset_folder:', cfg['dataset_folder'])
"

$python_cmd --parameters_file=/tmp/nibi_${job_name}_\${SLURM_JOB_ID}.json

echo '========================================'
echo "Finished: \$(date)"
echo '========================================'
EOF
)
    echo "  [$sbatch_id] $job_name  ($desc)"
    ((job_count++))
}

# DoReFaNet
submit_job "imgnet_dorefa" \
    "python3 -u src/Experiments/experiment_DoReFaNet.py" \
    "DoReFaNet ${EPOCHS}ep" \
    "./parameters_files/ImageNet/imagenet_convnext_doReFa_prod.json"

# TTQ
submit_job "imgnet_ttq" \
    "python3 -u src/Experiments/experiment_TTQ.py" \
    "TTQ ${EPOCHS}ep" \
    "./parameters_files/ImageNet/imagenet_convnext_TTQ_prod.json"

# pTTQ x3
for k_val in 0.5 1.0 1.2; do
    submit_job "imgnet_pttq_k${k_val}" \
        "python3 -u src/Experiments/experiment_pTTQ.py --k_override=$k_val" \
        "pTTQ k=$k_val ${EPOCHS}ep" \
        "./parameters_files/ImageNet/imagenet_convnext_pTTQ_prod.json"
done

# EMA-pTTQ x3
for k_val in 0.5 1.0 1.2; do
    submit_job "imgnet_emapttq_k${k_val}" \
        "python3 -u src/Experiments/experiment_pTTQ_experimental.py --k_override=$k_val" \
        "EMA-pTTQ k=$k_val ${EPOCHS}ep" \
        "./parameters_files/ImageNet/imagenet_convnext_experimental_prod.json"
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "Logs: ./logs/"
echo "============================================"
