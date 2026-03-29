#!/bin/bash
# ============================================================================
# ImageNet-1K ConvNeXt Quantized Methods — SLURM Submission Script
# ============================================================================
#
# Submits all quantized methods in parallel:
#   - DoReFaNet (1 job)
#   - TTQ (1 job)
#   - pTTQ (3 jobs: k=0.5, k=1.0, k=1.2)
#   - EMA-pTTQ (3 jobs: k=0.5, k=1.0, k=1.2)
#
# ImageNet-1K (1.28M images, 224x224) is ~20x slower per epoch than TinyImageNet.
# ConvNeXt has 58 quantizable layers. Full H100 needed for memory+speed.
#
# PREREQUISITE: FP baseline must be trained first!
#   ./sbatch/imagenet_fp.sh
#   (wait for completion)
#   ./sbatch/imagenet_quantized.sh
#
# Usage:
#   ./sbatch/imagenet_quantized.sh
#   ./sbatch/imagenet_quantized.sh --account def-myprof
#   ./sbatch/imagenet_quantized.sh --account def-myprof --gpus 4
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

# Verify FP model exists
FP_MODEL_DIR="./results/PROD_ImageNet_CONVNEXT_FP_OW_0/model/"
if [[ ! -d "$FP_MODEL_DIR" ]]; then
    echo "ERROR: FP model directory not found: $FP_MODEL_DIR"
    echo "Run ./sbatch/imagenet_fp.sh first and wait for completion."
    exit 1
fi

job_count=0

echo "============================================"
echo "ImageNet-1K ConvNeXt — Quantized Methods"
echo "============================================"

# ============================================================================
# HELPER: Submit a single job
# ============================================================================
submit_job() {
    local job_name="$1"
    local time_limit="$2"
    local python_cmd="$3"
    local desc="$4"

    local sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=./logs/${job_name}_%j.out
#SBATCH --error=./logs/${job_name}_%j.err
#SBATCH --time=$time_limit
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
echo "Job: $job_name"
echo "Config: $desc (${NUM_GPUS}x h100 DataParallel)"
echo "Time limit: $time_limit"
echo "Started: \$(date)"
echo '========================================'
nvidia-smi
export PYTHONPATH=\$PYTHONPATH:\$(pwd)

$python_cmd

echo '========================================'
echo "Finished: \$(date)"
echo '========================================'
EOF
)
    echo "  [$sbatch_id] $job_name  ($desc, $time_limit)"
    ((job_count++))
}

# ============================================================================
# DoReFaNet (1 job) — fastest quantized method
# ============================================================================
submit_job \
    "imgnet_dorefa" \
    "7-00:00:00" \
    "python3 src/Experiments/experiment_DoReFaNet.py --parameters_file=./parameters_files/ImageNet/imagenet_convnext_doReFa_prod.json" \
    "DoReFaNet ConvNeXt, 90ep"

# ============================================================================
# TTQ (1 job)
# ============================================================================
submit_job \
    "imgnet_ttq" \
    "7-00:00:00" \
    "python3 src/Experiments/experiment_TTQ.py --parameters_file=./parameters_files/ImageNet/imagenet_convnext_TTQ_prod.json" \
    "TTQ ConvNeXt, 90ep"

# ============================================================================
# pTTQ (3 jobs: k=0.5, k=1.0, k=1.2)
# ============================================================================
for k_val in 0.5 1.0 1.2; do
    submit_job \
        "imgnet_pttq_k${k_val}" \
        "7-00:00:00" \
        "python3 src/Experiments/experiment_pTTQ.py --parameters_file=./parameters_files/ImageNet/imagenet_convnext_pTTQ_prod.json --k_override=$k_val" \
        "pTTQ ConvNeXt k=$k_val, 90ep"
done

# ============================================================================
# EMA-pTTQ (3 jobs: k=0.5, k=1.0, k=1.2)
# ============================================================================
for k_val in 0.5 1.0 1.2; do
    submit_job \
        "imgnet_emapttq_k${k_val}" \
        "7-00:00:00" \
        "python3 src/Experiments/experiment_pTTQ_experimental.py --parameters_file=./parameters_files/ImageNet/imagenet_convnext_experimental_prod.json --k_override=$k_val" \
        "EMA-pTTQ ConvNeXt k=$k_val, 90ep"
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "Logs directory:       ./logs/"
echo "============================================"
