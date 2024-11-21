#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1           # Allocate all GPUs to the job
#SBATCH --mem=3000M
#SBATCH --time=1-00:00
#SBATCH --chdir=/scratch/sdmuhsin/Trainable-Pruned-Ternary-Quantization-with-Exponential-Moving-Average-Thresholds
#SBATCH --output=fmnist-TTQ-%N-%j.out

module load python/3.10
module load arrow/16.1.0
source ./env/bin/activate

echo "Environment loaded"
which python3
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python3 src/Experiments/experiment_TTQ.py --parameters_file=./parameters_files/FMNIST/fmnist_TTQ.json

echo "Starting run"
