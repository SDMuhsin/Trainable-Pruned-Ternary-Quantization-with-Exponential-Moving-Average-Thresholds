# Pruned trained ternary quantization for CNN models

## I) Introduction

GitHub repository for the sumbission of the paper *Extreme Compression of CNNs Using Exponential
Moving Average based Pruned Ternary Quantization*, currently under review, TNNLS.

## II) Configuration

To be able to run the different codes (Linux platforms), you need to start by running the following command:

    export PYTHONPATH="${PYTHONPATH}:path_to_code"

Then, you should install the different libraries needed to execute the different codes:

    pip install -r requirements.txt

## III) Proposed method

![image](https://github.com/SDMuhsin/Trainable-Pruned-Ternary-Quantization-with-Exponential-Moving-Average-Thresholds/blob/main/pttq_ema_flow.png) 

Our proposed method is a modification of [pTTQ](https://www.sciencedirect.com/science/article/pii/S0925231224009871) , which is inspired from TTQ ([Zhu et al. (2016)](https://arxiv.org/abs/1612.01064)) and consists of:
- **Pruning:** pruning is done before ternarization based on the weights's statistics and one weakly-differentiable pruning function with learnable parameters.
- **Ternarization**: the remaining positive weights are set to $1$ and the negatives ones to $-1$.
- **Scaling**: two full-precision scaling trainable parameters are assocaited to the ternary weights tensor, one for the positive weights $W_r$, and one for the negative ones, $W_l$.

Out contribution is as follows:
- Rather than using fluctuation prone instantaneous weight statistics, we use a global moving average based pruning mechanism
- We introduce a tunable parameter that allows the practitioner to directly control the degree of pruning

The paper is currently under review, for more information (and if you need to cite our work), please contact the author at ckp908@usask.ca

## IV) Code structure

This repository is structured using different folders:
- **data**: contains the different datasets (ESR and MNIST) used in the different experiments. At the beginning it empty, but when executing the first experiments, data will be downloaded in this folder.
- **parameters_files**: This folder contains json files with the parameters of different experiments.
- **results**: This folder contains the results of the different experiments. At the beginning it is empty, but when executing different experiments, results folders will be generated and saved here.
- **src**: This folder contains the different source codes. More precisely, it contains the source codes used to perform the different experiments in the paper. 

## V) Examples

All the experiments in the paper can be done executing the source codes in the root folder. The generated results will be stored in the folder *results* and can be plotted using different codes in *src/utils/*.

### a. Experiments.


- **Comparison with SOTA**: Use the codes *train_model_base.py*, *experiment_TTQ.py*, and *experiment_pTTQ.py* with the parameters files in *parameters_files/*. 
    -*train_model_base.py*: trains full precision models without quantization. The models generated by this experiment are the one that can then be fed to the quantization experiment's codes. Example (execution from the folder *src/Experiments/*): 
    
                            python train_model_base.py --parameters_file ./parameters_files/MNIST/mnist_FP.json
                            
    -*experiment_TTQ.py*: trains TTQ quantized models, using a pre-trained FP model.
    
                            python experiment_TTQ.py --parameters_file ./parameters_files/MNIST/mnist_TTQ.json
                            
    -*experiment_pTTQ.py*: trains pTTQ quantized models, using a pre-trained FP model. 
    
                            python experiment_pTTQ.py --parameters_file ./parameters_files/MNIST/mnist_pTTQ.json
    -*experiment_pTTQ_experimental.py*: trains pTTQ quantized models, using a pre-trained FP model.
    
                            python experiment_pTTQ.py --parameters_file ./parameters_files/MNIST/mnist_pTTQ.json                            
- **Ablation Studies on k and beta** : These are unique to EMA-pTTQ and triggering there are a matter specifying the required parameters in the corresponding parameters/DATASET/\* file.

    
### b. Plot results.

The results obtained in the previous experiments before, can be plotted and visualized using different codes in *src/utils/*:
- **getResultMetrics.py**: print average (across repetitions) best metrics (decided by best MCC across epochs) for sparsity, mcc and convergence
			   python getResultMetrics.py --paramters_pth_file=./results/MNIST_2D_CNN_FP/metrics/results*.pth 

- **getCompressionRate.py**: allows to get the compression rates and compression rates of a model with respect to another one.

                            python getCompressionRate.py --exp_folder_model_a ./results/MNIST_2D_CNN_FP/ --is_model_a_ternarized False --exp_folder_model_b ./results/MNIST_pTTQ/ --is_model_b_ternarized True

- **getEnergyConsumption.py**: computes the energy consumption of a model Model_Q with respect to a reference Model_FP.

                            python getEnergyConsumption.py --exp_results_folder_ref PATH/TO/MODEL_FP/ --exp_results_folder PATH/TO/MODEL_Q/
