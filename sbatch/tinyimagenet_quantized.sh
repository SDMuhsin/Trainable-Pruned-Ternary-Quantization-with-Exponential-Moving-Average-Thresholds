#!/bin/bash
# ============================================================================
# TinyImageNet ConvNeXt Quantized Methods — SLURM Submission Script
# ============================================================================
#
# Submits all quantized methods in parallel:
#   - DoReFaNet (1 job)
#   - TTQ (1 job)
#   - pTTQ (3 jobs: k=0.5, k=1.0, k=1.2)
#   - EMA-pTTQ (3 jobs: k=0.5, k=1.0, k=1.2)
#
# ConvNeXt is ~5-8x slower per epoch than ResNet-50 for quantized methods
# due to 58 quantizable layers with per-step threshold/EMA updates.
# Time limits reflect this (from prior ConvNeXt benchmarks).
#
# PREREQUISITE: FP baseline must be trained first!
#   ./sbatch/tinyimagenet_fp.sh
#   (wait for completion)
#   ./sbatch/tinyimagenet_quantized.sh
#
# Usage:
#   ./sbatch/tinyimagenet_quantized.sh
#   ./sbatch/tinyimagenet_quantized.sh --account def-myprof
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

# Verify FP model exists
FP_MODEL_DIR="./results/PROD_TinyImageNet_CONVNEXT_FP_OW_0/model/"
if [[ ! -d "$FP_MODEL_DIR" ]]; then
    echo "ERROR: FP model directory not found: $FP_MODEL_DIR"
    echo "Run ./sbatch/tinyimagenet_fp.sh first and wait for completion."
    exit 1
fi

job_count=0

echo "============================================"
echo "TinyImageNet ConvNeXt — Quantized Methods"
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
echo "Job: $job_name"
echo "Config: $desc"
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
# DoReFaNet (1 job) — ~17 min/ep on loaded GPU -> ~2d on 3g.40gb for 50ep
# ============================================================================
submit_job \
    "tinyimg_dorefa" \
    "2-00:00:00" \
    "python3 src/Experiments/experiment_DoReFaNet.py --parameters_file=./parameters_files/TinyImageNet/tinyimagenet_convnext_doReFa_prod.json" \
    "DoReFaNet ConvNeXt, 50ep"

# ============================================================================
# TTQ (1 job) — ~93 min/ep on loaded GPU -> ~5d on 3g.40gb for 50ep
# ============================================================================
submit_job \
    "tinyimg_ttq" \
    "5-00:00:00" \
    "python3 src/Experiments/experiment_TTQ.py --parameters_file=./parameters_files/TinyImageNet/tinyimagenet_convnext_TTQ_prod.json" \
    "TTQ ConvNeXt, 50ep"

# ============================================================================
# pTTQ (3 jobs: k=0.5, k=1.0, k=1.2) — similar to TTQ, ~5d
# ============================================================================
for k_val in 0.5 1.0 1.2; do
    submit_job \
        "tinyimg_pttq_k${k_val}" \
        "5-00:00:00" \
        "python3 src/Experiments/experiment_pTTQ.py --parameters_file=./parameters_files/TinyImageNet/tinyimagenet_convnext_pTTQ_prod.json --k_override=$k_val" \
        "pTTQ ConvNeXt k=$k_val, 50ep"
done

# ============================================================================
# EMA-pTTQ (3 jobs: k=0.5, k=1.0, k=1.2) — ~153 min/ep -> ~7d for 50ep
# ============================================================================
for k_val in 0.5 1.0 1.2; do
    submit_job \
        "tinyimg_emapttq_k${k_val}" \
        "7-00:00:00" \
        "python3 src/Experiments/experiment_pTTQ_experimental.py --parameters_file=./parameters_files/TinyImageNet/tinyimagenet_convnext_experimental_prod.json --k_override=$k_val" \
        "EMA-pTTQ ConvNeXt k=$k_val, 50ep"
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "Logs directory:       ./logs/"
echo "============================================"
