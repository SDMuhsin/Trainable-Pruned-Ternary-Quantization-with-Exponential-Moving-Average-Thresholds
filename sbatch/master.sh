#!/bin/bash

# Block 2: Run all other scripts
echo "Running other scripts..."
for dataset in cifar100 mnist kmnist cifar10 emnist fmnist stl10 svhn
do
    for script_type in doReFa experimental pTTQ TTQ
    do
        echo "Running ${dataset}_${script_type}.sh"
        sbatch ./sbatch/${dataset}_${script_type}.sh
    done
done


exit 1




# Block 1: Run all *_fp.sh scripts
echo "Running fp scripts..."
for dataset in cifar100 cifar10 emnist fmnist kmnist mnist stl10 svhn
do
    echo "Running ${dataset}_fp.sh"
    sbatch ./sbatch/${dataset}_fp.sh
done

exit 1
echo "All scripts have been executed."

exit 1


