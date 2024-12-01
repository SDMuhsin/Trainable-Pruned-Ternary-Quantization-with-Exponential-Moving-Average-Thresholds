#!/bin/bash

echo "Beginning k override sbatch scripts for CIFAR10"

# Iterate over k_override values from 0.1 to 2.0
echo "Submitting job with k_override = $k"
sbatch \
--nodes=1 \
--ntasks-per-node=1 \
--cpus-per-task=1 \
--gpus=1 \
--mem=8000M \
--time=3-00:00 \
--chdir=/scratch/sdmuhsin/Trainable-Pruned-Ternary-Quantization-with-Exponential-Moving-Average-Thresholds \
--output=cifar10-exp3-%N-%j.out \
--wrap="
    module load python/3.10
    module load arrow/16.1.0
    source ./env/bin/activate
    echo 'Environment loaded'
    which python3
    export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
    python3 src/Experiments/experiment_TTQ.py --parameters_file=./parameters_files/CIFAR10/cifar10_exp3.json
"

echo "All jobs submitted"
