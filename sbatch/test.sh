#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300M
#SBATCH --time=0-00:10
#SBATCH --chdir=/scratch/sdmuhsin/Trainable-Pruned-Ternary-Quantization-with-Exponential-Moving-Average-Thresholds
#SBATCH --output=test-%N-%j.out

module load python/3.10
module load arrow/16.1.0
source ./env/bin/activate

echo "Environment loaded"
which python3
export PYTHONPATH="$PYTHONPATH:$(pwd)"

echo "Starting run"
