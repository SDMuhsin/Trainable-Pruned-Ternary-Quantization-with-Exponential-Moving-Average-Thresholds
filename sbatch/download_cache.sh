#!/bin/bash
# ============================================================================
# Pre-download all HuggingFace models, datasets, and metrics to local cache
# ============================================================================
#
# GPU compute nodes on rorqual have NO internet access. Run this script on a
# LOGIN NODE before submitting any sbatch jobs.
#
# What this downloads:
#   - bert-base-uncased model (tokenizer, config, weights)
#   - GLUE datasets: cola, mrpc, rte, stsb, sst2, qnli
#   - GLUE evaluation metrics for each task
#
# After running this, the sbatch jobs use HF_DATASETS_OFFLINE=1 and
# TRANSFORMERS_OFFLINE=1 to read from cache without network access.
#
# For Tiny ImageNet: download is handled separately (not HuggingFace).
# See instructions at the bottom of this script.
#
# Usage:
#   ./sbatch/download_cache.sh
#
# ============================================================================

set -e

source ./env/bin/activate

export HF_HOME=$(pwd)/data
export TORCH_HOME=$(pwd)/data
mkdir -p $HF_HOME

echo "============================================"
echo "EMA-pTTQ — Download Cache"
echo "Cache directory: $HF_HOME"
echo "============================================"
echo ""

# ============================================================================
# BERT model
# ============================================================================

echo "=== Downloading bert-base-uncased ==="

python3 -c "
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

model_name = 'bert-base-uncased'

# Download and cache tokenizer
print(f'  Tokenizer...')
AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Download and cache model weights + config for each num_labels variant we use
for n_labels in [1, 2, 3]:
    print(f'  Model (num_labels={n_labels})...')
    config = AutoConfig.from_pretrained(model_name, num_labels=n_labels)
    AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

print('  Done: bert-base-uncased')
"

echo ""

# ============================================================================
# GLUE datasets
# ============================================================================

echo "=== Downloading GLUE datasets ==="

python3 -c "
from datasets import load_dataset

tasks = ['cola', 'mrpc', 'rte', 'stsb', 'sst2', 'qnli']
for task in tasks:
    print(f'  glue/{task}...')
    load_dataset('glue', task)
    print(f'  Done: glue/{task}')
"

echo ""

# ============================================================================
# Evaluation metrics
# ============================================================================

echo "=== Downloading evaluation metrics ==="

python3 -c "
import evaluate

# The GLUE script calls evaluate.load('glue', task_name) for each task.
# Pre-download the glue metric module (covers all subtasks).
tasks = ['cola', 'mrpc', 'rte', 'stsb', 'sst2', 'qnli']
for task in tasks:
    try:
        print(f'  glue/{task}...')
        evaluate.load('glue', task)
        print(f'  Done: glue/{task}')
    except Exception as e:
        print(f'  Warning: glue/{task}: {e}')
"

echo ""
echo "============================================"
echo "All HuggingFace downloads complete."
echo "Cache directory: $HF_HOME"
echo ""
echo "--- Tiny ImageNet ---"
echo "Tiny ImageNet is NOT a HuggingFace dataset. To prepare it:"
echo "  1. On a machine with internet, run:"
echo "       python3 -c \"from src.DataManipulation.tinyimagenet_data import download_and_prepare_tinyimagenet; download_and_prepare_tinyimagenet('./data/TinyImageNet/')\""
echo "  2. Then rsync the prepared data to the cluster:"
echo "       rsync -av ./data/TinyImageNet/ USER@rorqual.calculquebec.ca:PATH_TO_PROJECT/data/TinyImageNet/"
echo "  The folder must contain: data/TinyImageNet/tiny-imagenet-200/{train,val}/.prepared"
echo "============================================"
