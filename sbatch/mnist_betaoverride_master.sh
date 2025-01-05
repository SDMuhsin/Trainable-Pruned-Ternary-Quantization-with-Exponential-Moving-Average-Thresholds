#!/bin/bash

echo "Beginning beta override sbatch scripts for MNIST"

# Iterate over k_override values from 0.1 to 2.0
for beta in $(seq 0.1 0.1 1.0)
do
    echo "Submitting job with beta = $beta"
    sbatch \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=2 \
        --gpus=1 \
        --mem=6000M \
        --time=2-00:00 \
        --chdir=/scratch/sdmuhsin/Trainable-Pruned-Ternary-Quantization-with-Exponential-Moving-Average-Thresholds \
        --output=mnist-beta${beta}-%N-%j.out \
        --wrap="
            module load python/3.10
            module load arrow/16.1.0
            source ./env/bin/activate
            echo 'Environment loaded'
            which python3
            export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
            python3 src/Experiments/experiment_pTTQ_experimental.py --parameters_file=./parameters_files/MNIST/mnist_experimental.json --k_override=1.0 --beta=$beta
        "
done

echo "All jobs submitted"
