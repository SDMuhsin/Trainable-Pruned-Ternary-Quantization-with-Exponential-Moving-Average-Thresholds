#!/bin/bash
# ============================================================================
# BERT GLUE Ternary Quantization — SLURM Submission Script
# ============================================================================
#
# Submits Mo5 (median-of-5, seeds 41-45) experiments for all methods and tasks:
#   - FP baseline (6 tasks)
#   - TTQ (6 tasks)
#   - pTTQ (6 tasks × 3 k values)
#   - EMA-pTTQ (6 tasks × 3 k values)
#
# All results are written to a shared thread-safe CSV: ./results/glue_ternary.csv
#
# Usage:
#   ./sbatch/glue_all.sh
#   ./sbatch/glue_all.sh --account def-myprof
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

# ============================================================================
# CONFIGURATION
# ============================================================================

tasks=("cola" "mrpc" "rte" "stsb" "sst2" "qnli")
k_values=("0.5" "1.0" "1.2")

# Stable hyperparameters (validated on CoLA and MRPC)
WARMUP_EPOCHS=2
QUANT_EPOCHS=15
FP_EPOCHS=3
LR_QUANT="5e-6"
LR_THRESH="0.01"
OPT_THRESH="sgd"
SCHEDULER="constant_with_warmup"

job_count=0

# ============================================================================
# TIME LIMITS PER TASK
# ============================================================================
# Based on dataset sizes and estimated Mo5 runtimes on h100_3g.40gb.
# Quantized: 5 seeds × (2 WU + 15 quant) = 85 epochs
# FP: 5 seeds × 3 epochs = 15 epochs
#
# Task       Train    Steps/ep  Quant Est.   FP Est.
# CoLA       8,551    267       ~12h         ~1h
# MRPC       3,668    115       ~5h          ~0.5h
# RTE        2,490    78        ~4h          ~0.3h
# STS-B      5,749    180       ~8h          ~0.5h
# SST-2      67,349   2,105     ~86h         ~5h
# QNLI       104,743  3,273     ~134h        ~8h

get_quant_time() {
    case $1 in
        cola)  echo "24:00:00" ;;
        mrpc)  echo "12:00:00" ;;
        rte)   echo "8:00:00"  ;;
        stsb)  echo "18:00:00" ;;
        sst2)  echo "5-00:00:00" ;;
        qnli)  echo "7-00:00:00" ;;
    esac
}

get_fp_time() {
    case $1 in
        cola)  echo "3:00:00"  ;;
        mrpc)  echo "2:00:00"  ;;
        rte)   echo "1:00:00"  ;;
        stsb)  echo "2:00:00"  ;;
        sst2)  echo "12:00:00" ;;
        qnli)  echo "18:00:00" ;;
    esac
}

# GPU allocation: 3g.40gb for all tasks (sufficient VRAM, good compute)
GPU_TYPE="h100_3g.40gb:1"
GPU_MEM="32000M"

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
#SBATCH --gpus=$GPU_TYPE
#SBATCH --mem=$GPU_MEM
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
echo "Cache: \$HF_HOME"
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

echo "============================================"
echo "BERT GLUE Ternary Quantization Suite"
echo "============================================"
echo "Tasks:      ${tasks[*]}"
echo "Methods:    FP, TTQ, pTTQ, EMA-pTTQ"
echo "k values:   ${k_values[*]}"
echo "Seeds:      41,42,43,44,45 (Mo5)"
echo "Scheduler:  $SCHEDULER"
echo "============================================"
echo ""

# ============================================================================
# FP BASELINES (6 jobs)
# ============================================================================
echo "--- FP Baselines ---"
for task in "${tasks[@]}"; do
    time_limit=$(get_fp_time "$task")
    submit_job \
        "glue_fp_${task}" \
        "$time_limit" \
        "python3 -u src/Experiments/train_glue_quantized.py --task_name $task --method fp --epochs $FP_EPOCHS --device cuda --seeds 41,42,43,44,45" \
        "FP $task ${FP_EPOCHS}ep"
done
echo ""
# ============================================================================
# TTQ (6 jobs)
# ============================================================================
echo "--- TTQ ---"
for task in "${tasks[@]}"; do
    time_limit=$(get_quant_time "$task")
    submit_job \
        "glue_ttq_${task}" \
        "$time_limit" \
        "python3 -u src/Experiments/train_glue_quantized.py --task_name $task --method ttq --warmup_epochs $WARMUP_EPOCHS --epochs $QUANT_EPOCHS --scheduler_type $SCHEDULER --smart_initial_scales --device cuda --seeds 41,42,43,44,45" \
        "TTQ $task ${WARMUP_EPOCHS}WU+${QUANT_EPOCHS}ep"
done
echo ""

# ============================================================================
# pTTQ (6 tasks × 3 k values = 18 jobs)
# ============================================================================
echo "--- pTTQ ---"
for task in "${tasks[@]}"; do
    time_limit=$(get_quant_time "$task")
    for k_val in "${k_values[@]}"; do
        submit_job \
            "glue_pttq_${task}_k${k_val}" \
            "$time_limit" \
            "python3 -u src/Experiments/train_glue_quantized.py --task_name $task --method pttq --warmup_epochs $WARMUP_EPOCHS --epochs $QUANT_EPOCHS --lr_quant $LR_QUANT --lr_thresh $LR_THRESH --optimizer_thresh $OPT_THRESH --scheduler_type $SCHEDULER --k $k_val --smart_initial_scales --device cuda --seeds 41,42,43,44,45" \
            "pTTQ $task k=$k_val ${WARMUP_EPOCHS}WU+${QUANT_EPOCHS}ep"
    done
done
echo ""

# ============================================================================
# EMA-pTTQ (6 tasks × 3 k values = 18 jobs)
# ============================================================================
echo "--- EMA-pTTQ ---"
for task in "${tasks[@]}"; do
    time_limit=$(get_quant_time "$task")
    for k_val in "${k_values[@]}"; do
        submit_job \
            "glue_emapttq_${task}_k${k_val}" \
            "$time_limit" \
            "python3 -u src/Experiments/train_glue_quantized.py --task_name $task --method ema_pttq --warmup_epochs $WARMUP_EPOCHS --epochs $QUANT_EPOCHS --lr_quant $LR_QUANT --lr_thresh $LR_THRESH --optimizer_thresh $OPT_THRESH --scheduler_type $SCHEDULER --k $k_val --smart_initial_scales --device cuda --seeds 41,42,43,44,45" \
            "EMA-pTTQ $task k=$k_val ${WARMUP_EPOCHS}WU+${QUANT_EPOCHS}ep"
    done
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "Results CSV:          ./results/glue_ternary.csv"
echo "Logs directory:       ./logs/"
echo "============================================"
