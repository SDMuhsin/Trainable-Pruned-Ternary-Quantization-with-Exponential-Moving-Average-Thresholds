#!/bin/bash

echo "Beginning EXP3 sbatch scripts for MNIST"

sbatch \
	--nodes=1 \
	--ntasks-per-node=1 \
	--cpus-per-task=1 \
	--gpus=1 \
	--mem=6000M \
	--time=2-00:00 \
	--chdir=/scratch/sdmuhsin/Trainable-Pruned-Ternary-Quantization-with-Exponential-Moving-Average-Thresholds \
	--output=mnist-exp3-pttq-%N-%j.out \
	--wrap="
	    module load python/3.10
	    module load arrow/16.1.0
	    source ./env/bin/activate
	    echo 'Environment loaded'
	    which python3
	    export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
	    python3 src/Experiments/experiment_pTTQ.py --parameters_file=./parameters_files/MNIST/mnist_exp3_pTTQ.json
	"

echo "All jobs submitted"
