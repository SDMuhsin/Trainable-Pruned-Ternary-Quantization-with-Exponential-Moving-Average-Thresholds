#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1           # Allocate all GPUs to the job
#SBATCH --mem=3000M
#SBATCH --time=1-00:00
#SBATCH --chdir=/scratch/sdmuhsin/DELU
#SBATCH --output=fmnist-%N-%j.out

module load python/3.10
module load arrow/16.1.0
source ./env/bin/activate

echo "Environment loaded"
which python3

python3 src/Experiments/train_model_base.py --parameters_file=./parameters/KMNIST/kmnist_FP.json

echo "Starting run"
