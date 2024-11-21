#!/bin/sh

echo "Beginning k override sbatch scripts for MNIST"

# Iterate over k_override values from 0.1 to 2.0
for k in $(seq 0.1 0.1 0.2)
do
    echo "Submitting job with k_override = $k"
    sbatch --wrap "sh sbatch/sbatch_base.sh && python3 src/Experiments/experiment_pTTQ_experimental.py --parameters_file=./parameters_files/MNIST/mnist_experimental.json --k_override=$k"
done

echo "All jobs submitted"
