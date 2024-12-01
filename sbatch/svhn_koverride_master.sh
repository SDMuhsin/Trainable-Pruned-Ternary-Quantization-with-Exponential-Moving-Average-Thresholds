#!/bin/bash

echo "Beginning k override sbatch scripts for SVHN"

# Iterate over k_override values from 0.1 to 2.0
for k in $(seq 0.1 0.1 2.0)
do
    echo "Submitting job with k_override = $k"
    sbatch \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=1 \
        --gpus=1 \
        --mem=6000M \
        --time=2-00:00 \
        --chdir=/scratch/sdmuhsin/Trainable-Pruned-Ternary-Quantization-with-Exponential-Moving-Average-Thresholds \
        --output=svhn-k${k}-%N-%j.out \
        --wrap="
            module load python/3.10
            module load arrow/16.1.0
            source ./env/bin/activate
            echo 'Environment loaded'
            which python3
            export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
            python3 src/Experiments/experiment_pTTQ_experimental.py --parameters_file=./parameters_files/SVHN/svhn_experimental.json --k_override=$k
        "
done

echo "All jobs submitted"
