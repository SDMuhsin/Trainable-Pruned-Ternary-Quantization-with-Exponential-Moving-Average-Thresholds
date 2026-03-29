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
# Nibi GPU nodes: 8x H100 80GB per node, Intel Xeon 6 Granite Rapids
# Request format: --gres=gpu:h100:N  (full GPUs, no MIG slices)
#
# Quantized methods have per-layer custom gradient steps that don't
# parallelize across GPUs, so the forward pass is the main beneficiary
# of multi-GPU. 4 GPUs is the sweet spot for throughput/cost.
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
    echo "Usage: $0 --account def-myprof [--gpus $NUM_GPUS] [--epochs $EPOCHS]"
    exit 1
fi

# Scale resources per GPU
CPUS=$((NUM_GPUS * 12))
MEM=$((NUM_GPUS * 30000))

mkdir -p ./logs ./results

# Verify FP model exists
FP_MODEL_DIR="./results/PROD_ImageNet_CONVNEXT_FP_OW_0/model/"
if [[ ! -d "$FP_MODEL_DIR" ]]; then
    echo "ERROR: FP model directory not found: $FP_MODEL_DIR"
    echo "Run ./sbatch/nibi_imagenet_fp.sh first and wait for completion."
    exit 1
fi

job_count=0

echo "============================================"
echo "ImageNet-1K ConvNeXt — Quantized Methods (Nibi)"
echo "  GPUs per job: ${NUM_GPUS}x H100 80GB"
echo "  Epochs: ${EPOCHS}"
echo "  Time limit: 2-00:00:00"
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
echo "Job: $job_name"
echo "Config: $desc"
echo "Cluster: Nibi — ${NUM_GPUS}x H100 (DataParallel)"
echo "Epochs: ${EPOCHS}"
echo "Started: \$(date)"
echo '========================================'
nvidia-smi

# Write a temporary config with the requested epoch count
python3 -c "
import json
with open('$config_file') as f:
    cfg = json.load(f)
cfg['nb_epochs'] = ${EPOCHS}
with open('/tmp/nibi_${job_name}_\${SLURM_JOB_ID}.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Config written with nb_epochs=${EPOCHS}')
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

# ============================================================================
# DoReFaNet (1 job)
# ============================================================================
submit_job \
    "imgnet_dorefa" \
    "python3 -u src/Experiments/experiment_DoReFaNet.py" \
    "DoReFaNet ConvNeXt, ${EPOCHS}ep, ${NUM_GPUS}x H100" \
    "./parameters_files/ImageNet/imagenet_convnext_doReFa_prod.json"

# ============================================================================
# TTQ (1 job)
# ============================================================================
submit_job \
    "imgnet_ttq" \
    "python3 -u src/Experiments/experiment_TTQ.py" \
    "TTQ ConvNeXt, ${EPOCHS}ep, ${NUM_GPUS}x H100" \
    "./parameters_files/ImageNet/imagenet_convnext_TTQ_prod.json"

# ============================================================================
# pTTQ (3 jobs: k=0.5, k=1.0, k=1.2)
# ============================================================================
for k_val in 0.5 1.0 1.2; do
    submit_job \
        "imgnet_pttq_k${k_val}" \
        "python3 -u src/Experiments/experiment_pTTQ.py --k_override=$k_val" \
        "pTTQ ConvNeXt k=$k_val, ${EPOCHS}ep, ${NUM_GPUS}x H100" \
        "./parameters_files/ImageNet/imagenet_convnext_pTTQ_prod.json"
done

# ============================================================================
# EMA-pTTQ (3 jobs: k=0.5, k=1.0, k=1.2)
# ============================================================================
for k_val in 0.5 1.0 1.2; do
    submit_job \
        "imgnet_emapttq_k${k_val}" \
        "python3 -u src/Experiments/experiment_pTTQ_experimental.py --k_override=$k_val" \
        "EMA-pTTQ ConvNeXt k=$k_val, ${EPOCHS}ep, ${NUM_GPUS}x H100" \
        "./parameters_files/ImageNet/imagenet_convnext_experimental_prod.json"
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "All jobs: 2-day limit, ${NUM_GPUS}x H100, ${EPOCHS} epochs"
echo "Logs: ./logs/"
echo "============================================"
