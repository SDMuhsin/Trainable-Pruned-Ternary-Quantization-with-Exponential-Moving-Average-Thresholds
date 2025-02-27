#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1           # Allocate all GPUs to the job
#SBATCH --mem=6000M
#SBATCH --time=2-00:00
#SBATCH --chdir=/scratch/sdmuhsin/Trainable-Pruned-Ternary-Quantization-with-Exponential-Moving-Average-Thresholds
#SBATCH --output=cifar100-experimental-%N-%j.out

module load python/3.10
module load arrow/16.1.0
source ./env/bin/activate

echo "Environment loaded"
which python3
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python3 src/Experiments/experiment_pTTQ_experimental.py --parameters_file=./parameters_files/CIFAR100/cifar100_experimental.json

echo "Starting run"
