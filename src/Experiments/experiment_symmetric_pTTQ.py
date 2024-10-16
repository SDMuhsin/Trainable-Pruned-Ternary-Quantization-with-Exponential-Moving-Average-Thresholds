#!/usr/bin/env python3
"""
    Compress a pre-trained model using atuomated pruning (Manessi et al. 2019)
    with TTQ.

    Options:
    --------
    --parameters_file: str
        Path to a file containing the parameters of the experiment.
        This files are usually located in /hits_signal_learning/parameters_files/model_compression/
"""
import os
import json
import shutil
import pickle
import argparse
from tqdm import tqdm

import random

import numpy as np
from math import floor
import matplotlib as mpl
from datetime import datetime

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

import torch
import torch.nn as nn
from torchsummary import summary
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch.nn.utils.prune as prune
from pydub import AudioSegment

from labml_nn.optimizers import noam

from src.Experiments.experiment_TTQ import Experiment as ExperimentTTQ

from src.utils.GCE import GeneralizedCrossEntropy
from src.utils.model_compression import approx_weights, approx_weights_fc, pruning_function_asymmetric_manessi

from src.Models.CNNs.time_frequency_simple_CNN import TimeFrequency2DCNN # Network used for training
from src.Models.Transformers.Transformer_Encoder_RawAudioMultiChannelCNN import TransformerClassifierMultichannelCNN


#==============================================================================#
#======================== Defining the experiment class ========================#
#==============================================================================#
class Experiment(ExperimentTTQ):
    def __init__(self, parameters_exp):
        """
            Compress a pre-trained model using pruned TTQ.

            Arguments:
            ----------
            parameters_exp: dict
                Dictionary containing the parameters of the experiment:
                    * exp_id: str, name of the experiment.
                    * feature_type
        """
        # Parent constructor
        super().__init__(parameters_exp)

        # Some attributes
        # Optimize alpha bool
        if ('optimize_alpha' not in parameters_exp):
            parameters_exp['optimize_alpha'] = False
        self.optimize_alpha = parameters_exp['optimize_alpha']
        # Alpha value
        if ('alpha' not in parameters_exp):
            parameters_exp['alpha'] = 1e2
        self.alpha = parameters_exp['alpha']
        # Learning rates (other than the main ones)
        if ('lr_wn_wp' not in parameters_exp):
            parameters_exp['lr_wn_wp'] = self.lr
        self.lr_wn_wp = parameters_exp['lr_wn_wp']
        if ('lr_thresh' not in parameters_exp):
            parameters_exp['lr_thresh'] = self.lr
        self.lr_thresh = parameters_exp['lr_thresh']
        if ('lr_alpha' not in parameters_exp):
            parameters_exp['lr_alpha'] = self.lr
        self.lr_alpha = parameters_exp['lr_alpha']

        # Initial Threshold hyper-parameter for the pruning function
        if ('init_t' not in parameters_exp):
            parameters_exp['init_t'] = 5e-1
        self.init_t = parameters_exp['init_t']

        # Pruning function
        if ('pruning_function_type' not in parameters_exp):
            parameters_exp['pruning_function_type'] = 'manessi_asymmetric'
        self.pruning_function_type = parameters_exp['pruning_function_type']
        if (self.pruning_function_type.lower() == 'manessi_asymmetric'):
            self.pruning_function = pruning_function_asymmetric_manessi
        else:
            raise ValueError("Pruning function {} is not valid".format(self.pruning_function_type))

        # Parameters of the exp
        self.parameters_exp = parameters_exp


    # Quantization function
    def quantize(self, kernel, w_p, w_n):
        """
        Function from inspired from https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py

        Return quantized weights of a layer.
        Only possible values of quantized weights are: {zero, w_p, -w_n}.
        """
        # WARNING: TODO VERIFY IF IT IS CORRECT !!!!
        # Getting the pruned kernel
        pruned_kernel = pruning_function_asymmetric_manessi(kernel, alpha=self.alpha, a=self.t, b=self.t)
        A = (pruned_kernel > 0).float()
        B = (pruned_kernel < 0).float()
        return w_p*A + (-w_n*B)

    # Gradients computation
    def get_grads(self, kernel_grad, kernel, w_p, w_n):
        """
        Function from: https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py

        Arguments:
            kernel_grad: gradient with respect to quantized kernel.
            kernel: corresponding full precision kernel.
            w_p, w_n: scaling factors.
            t: hyperparameter for quantization.
        Returns:
            1. gradient for the full precision kernel.
            2. gradient for w_p.
            3. gradient for w_n.
            4. gradient for a
            5. gradient for b
            6. gradient for alpha
        """
        # Grads of w_p and w_n
        pruned_kernel = pruning_function_asymmetric_manessi(kernel, alpha=self.alpha, a=self.t, b=self.t)
        A = (pruned_kernel > 0).float()
        B = (pruned_kernel < 0).float()
        c = torch.ones(pruned_kernel.size()).to(self.device) - A - B
        grad_fp_kernel = w_p*A*kernel_grad + w_n*B*kernel_grad + 1.0*c*kernel_grad
        grad_wp = (A*kernel_grad).sum()
        grad_wn = (B*kernel_grad).sum()

        # Grads of w_p and w_n
        if (self.pruning_function_type == 'manessi_asymmetric'):
            grad_t = (
                        -torch.heaviside((kernel-self.t).float(),  torch.tensor([0]).float().to(self.device) )\
                        +torch.heaviside((-kernel-self.t).float(),  torch.tensor([0]).float().to(self.device) )\
                        + torch.nn.functional.sigmoid(self.alpha*(kernel-self.t))\
                        - torch.nn.functional.sigmoid(self.alpha*(-kernel-self.t))\
                        - self.alpha*self.t*torch.nn.functional.sigmoid(self.alpha*(kernel-self.t))*(1-torch.nn.functional.sigmoid(self.alpha*(kernel-self.t)))
                        + self.alpha*self.t*torch.nn.functional.sigmoid(self.alpha*(-kernel-self.t))*(1-torch.nn.functional.sigmoid(self.alpha*(-kernel-self.t)))
                     ).sum()


            # Grads of alpha
            grad_alpha = (self.t*(kernel-self.t)*torch.nn.functional.sigmoid(self.alpha*(kernel-self.t))*(1-torch.nn.functional.sigmoid(self.alpha*(kernel-self.t)))\
                        + self.t*(kernel+self.t)*torch.nn.functional.sigmoid(self.alpha*(-kernel-self.t))*(1-torch.nn.functional.sigmoid(self.alpha*(-kernel-self.t)))).sum()
        else:
            raise NotImplementedError("Gradient computation for pruning function {} is not implemented yet.".format(self.pruning_function_type))
        return grad_fp_kernel, grad_wp, grad_wn, grad_t, grad_alpha

    def initial_alpha(self, kernel):
        return self.alpha

    def initial_thresholds(self, kernel):
        return self.init_t

    def createOptimizer(self, model_params_dict):
        """
            Creation of the optimizer(s)
        """
        # Optimizer for the model parameters
        self.optimizer = torch.optim.Adamax([model_params_dict[group_name] for group_name in model_params_dict], lr=self.lr)
        if (self.model_type.lower() != 'transformer'):
            # Creating the learning rate scheduler for the global optimizer
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',\
                                                               factor=0.1, patience=5,\
                                                               threshold=1e-4, threshold_mode='rel',\
                                                               cooldown=0, min_lr=0, eps=1e-08, verbose=False)

        # Copy the full precision weights of the parameters that are going to be
        # quantized (so not all the FP weights)
        kernels_to_quantize_fp_copy = [ Variable(kernel.data.clone(), requires_grad=True) for kernel in model_params_dict['ToQuantize']['params']]

        # Scaling factors for each quantized layer
        initial_scaling_factors = []

        # Initial thresholds
        initial_thresh = []

        # Initial alpha values
        if (self.optimize_alpha):
            initial_alphas = []

        # Kernels to be quantized
        kernels_to_quantize = [kernel for kernel in model_params_dict['ToQuantize']['params']]

        # Initial Quantization
        for k, k_fp in zip(kernels_to_quantize, kernels_to_quantize_fp_copy):
            # Getting the initial scaling factors
            w_p_initial, w_n_initial = self.initial_scales(k_fp.data)
            initial_scaling_factors += [(w_p_initial, w_n_initial)]

            # Getting the initial thresholds
            self.t = self.initial_thresholds(k_fp.data)
            initial_thresh += [(self.t)]

            # Getting the initial alpha
            if (self.optimize_alpha):
                alpha_initial = self.initial_alpha(k_fp.data)
                initial_alphas += [alpha_initial]
                self.alpha = alpha_initial

            # Doing quantization
            k.data = self.quantize(k_fp.data, w_p_initial, w_n_initial)

        # Getting the optimizers for the FP kernels and the scaling factors
        # FP kernels
        self.optimizer_fp = torch.optim.Adamax(kernels_to_quantize_fp_copy, lr=self.lr)
        # Scaling factors
        self.optimizer_sf = torch.optim.Adamax(
                                            [Variable(torch.FloatTensor([w_p, w_n]).to(self.device), requires_grad=True)
                                             for w_p, w_n in initial_scaling_factors],
                                            lr=self.lr_wn_wp
                                         )
        # Thresholds
        self.optimizer_t = torch.optim.Adamax(
                                            [Variable(torch.FloatTensor([t]).to(self.device), requires_grad=True)
                                             for t in initial_thresh],
                                            lr=self.lr_thresh
                                         )
        self.sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_t, T_max=10, eta_min=1e-7, last_epoch=-1, verbose=True)


        # Alpha value
        if (self.optimize_alpha):
            self.optimizer_alpha = torch.optim.Adamax(
                                                [Variable(torch.FloatTensor([alpha]).to(self.device), requires_grad=True)
                                                 for alpha in initial_alphas],
                                                lr=self.lr_alpha
                                             )
            # Creating learning rate scheduler for the alpha values
            self.sched_alpha = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_alpha, T_max=10, eta_min=1e-7, last_epoch=-1, verbose=True)
        else:
            self.optimizer_alpha = None


    def apply_lr_sched(self, mean_loss_val_fixed_epoch=None):
        """
            Applies the learning rate scheduler(s)
        """
        # Applying the generic lr scheduler strategy
        super().apply_lr_sched(mean_loss_val_fixed_epoch)

        # Applying lr scheduler for the thresholds
        self.sched_t.step()

        # Applying lr scheduler for the alpha values
        if (self.optimize_alpha):
            self.sched_alpha.step()


    def optimize_step(self, loss_value):
        """
            Simple optimization step
        """
        # Zero grad for all optimizers
        self.optimizer.zero_grad()
        self.optimizer_fp.zero_grad()
        self.optimizer_sf.zero_grad()
        self.optimizer_t.zero_grad()
        if (self.optimize_alpha):
            self.optimizer_alpha.zero_grad()

        # Gradients for the quantized model
        loss_value.backward()

        # Gett the quantized kernels
        quantized_kernels = self.optimizer.param_groups[1]['params']

        # Get the FP copies of the original kernels
        fp_kernels = self.optimizer_fp.param_groups[0]['params']

        # Getting the scaling factors of the kernels
        scaling_factors = self.optimizer_sf.param_groups[0]['params']

        # Getting the thresholds
        thresholds = self.optimizer_t.param_groups[0]['params']

        # Getting the alpha value if it is optimized
        if (self.optimize_alpha):
            alphas = self.optimizer_alpha.param_groups[0]['params']

        for i in range(len(quantized_kernels)):
            # Current quantized kernel
            k = quantized_kernels[i]

            # Getting the FP version of the current kernel
            k_fp = fp_kernels[i]

            # Getting the scaling factors for the quantized kernel
            f = scaling_factors[i]
            w_p, w_n = f.data[0], f.data[1]

            # Getting the thresholds used for pruning
            t = thresholds[i]
            self.t = t.data[0]

            # Getting the alpha values used for pruning
            if (self.optimize_alpha):
                alpha_val = alphas[i]
                self.alpha = alpha_val.data[0]

            # Getting the gradients
            k_fp_grad, w_p_grad, w_n_grad, t_grad, alpha_grad = self.get_grads(k.grad.data, k_fp.data, w_p, w_n)

            # Gradient for the full precision kernels
            k_fp.grad = Variable(k_fp_grad)

            # The quantized kernels are not updated (they are computed from the FP kernels)
            k.grad.data.zero_()

            # Gradient for the scaling factors
            f.grad = Variable(torch.FloatTensor([w_p_grad, w_n_grad]).to(self.device))

            # Gradient for the thresholds
            t.grad = Variable(torch.FloatTensor([t_grad]).to(self.device))

            # Gradient for the alpha value
            if (self.optimize_alpha):
                alpha_val.grad = Variable(torch.FloatTensor([alpha_grad]).to(self.device))

        # Update all the parameters that should not be quantized (usually the first and last
        # layers, as well as the batch norm parameters)
        self.optimizer.step()

        # Update the full precision kernels
        self.optimizer_fp.step()

        # Updating the scaling factor parameters
        self.optimizer_sf.step()

        # Updating the threshold parameters
        self.optimizer_t.step()

        # Updating the alpha parameter
        if (self.optimize_alpha):
            self.optimizer_alpha.step()

        # Ensuring that the values of the thresholds and alphas are positive using a clamp funcion
        for i in range(len(quantized_kernels)):
            # Thresholds
            t = thresholds[i]
            # Forcing positive values if exponentiation is not in function !!!!!!!
            if ('_exp' not in self.pruning_function_type.lower()) and ('attq' not in self.pruning_function_type.lower()):
                t.data[0] = torch.clamp(t.data[0], min=1e-5)

            # Ensuring that the value of alpha is positive using a clamp funtion
            if (self.optimize_alpha):
                # Current quantized kernel i
                alpha_val = alphas[i]
                # Forcing positive values if exponentiation is not in function !!!!!!!
                if ('_exp' not in self.pruning_function_type.lower()):
                    alpha_val.data[0] = torch.clamp(alpha_val.data[0], min=1e-8)

        # Quantize the updated full precision kernels
        for i in range(len(quantized_kernels)):
            # Current quantized kernel
            k = quantized_kernels[i]

            # Getting the FP version of the current kernel
            k_fp = fp_kernels[i]

            # Getting the scaling factors for the quantized kernel
            f = scaling_factors[i]
            w_p, w_n = f.data[0], f.data[1]

            # Getting the thresholds used for pruning
            t = thresholds[i]
            self.t = t.data[0]

            # Getting the alpha values used for pruning
            if (self.optimize_alpha):
                alpha_val = alphas[i]
                self.alpha = alpha_val.data[0]

            k.data = self.quantize(k_fp.data, w_p, w_n)

    def gridSearch(self):
        """
            Does a grid search for some hyper-parameters
        """
        # Defining the values of the parameters to test
        lr_values = [self.lr]
        lr_wn_wp_values = [self.lr_wn_wp]
        init_t_values = [1.0, 0.5]
        lr_thresh_values = [self.lr_thresh]
        alpha_values = [2e2, 9e1]
        lr_alpha_values = [self.lr_alpha]

        # Iterating over the different values of the hyper-parameters
        base_results_folder = self.results_folder
        for lr in lr_values:
            for lr_wn_wp in lr_wn_wp_values:
                for init_x in init_x_values:
                        for init_y in init_y_values:
                            for lr_thresh in lr_thresh_values:
                                for alpha in alpha_values:
                                    for lr_alpha in lr_alpha_values:
                                        # Updating the hyper-paramet of the experiment
                                        self.lr = lr
                                        self.lr_wn_wp = lr_wn_wp
                                        self.init_t = init_t
                                        self.lr_thresh = lr_thresh
                                        self.alpha = alpha
                                        self.lr_alpha = lr_alpha

                                        # Creating the datasets folder
                                        current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
                                        os.mkdir(base_results_folder + '/LR-{}_LRWNWP-{}_INITT-{}_LRTHRESH-{}_ALPHA-{}_LRALPHA-{}_{}/'.format(self.lr, self.lr_wn_wp, self.init_t, self.lr_thresh, self.alpha, self.lr_alpha, current_datetime))
                                        os.mkdir(base_results_folder + '/LR-{}_LRWNWP-{}_INITT-{}_LRTHRESH-{}_ALPHA-{}_LRALPHA-{}_{}/model/'.format(self.lr, self.lr_wn_wp, self.init_t, self.lr_thresh, self.alpha, self.lr_alpha, current_datetime))
                                        os.mkdir(base_results_folder + '/LR-{}_LRWNWP-{}_INITT-{}_LRTHRESH-{}_ALPHA-{}_LRALPHA-{}_{}/metrics/'.format(self.lr, self.lr_wn_wp, self.init_t, self.lr_thresh, self.alpha, self.lr_alpha, current_datetime))
                                        self.results_folder = base_results_folder + '/LR-{}_LRWNWP-{}_INITT-{}_LRTHRESH-{}_ALPHA-{}_LRALPHA-{}_{}/'.format(self.lr, self.lr_wn_wp, self.init_t, self.lr_thresh, self.alpha, self.lr_alpha, current_datetime)

                                        # Training
                                        self.holdout_train()

        self.results_folder = base_results_folder



#==============================================================================#
#================================ Main Function ================================#
#==============================================================================#
def main():
    print("\n\n==================== Beginning of the experiment ====================\n\n")
    #==========================================================================#
    # Fixing the random seed
    seed = 42
    random.seed(seed) # For reproducibility purposes
    np.random.seed(seed) # For reproducibility purposes
    torch.manual_seed(seed) # For reproducibility purposes
    if torch.cuda.is_available(): # For reproducibility purposes
        torch.cuda.manual_seed_all(seed)

    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    default_parameters_file = "../../parameters_files/MNIST/mnist_pruning_ttq_symmetric.json"
    ap.add_argument('--parameters_file', default=default_parameters_file, help="Parameters for the experiment", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    parameters_file = args['parameters_file']
    with open(parameters_file) as jf:
        parameters_exp = json.load(jf)

    # Grid search parameter in the parameters file
    if ('doGridSearch' not in parameters_exp):
        parameters_exp['doGridSearch'] = False
    doGridSearch = parameters_exp['doGridSearch']

    #==========================================================================#
    # Creating an instance of the experiment
    exp = Experiment(parameters_exp)

    # Creating directory to save the results
    inc = 0
    current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    resultsFolder = '../../results/' + parameters_exp['exp_id'] + '_' + current_datetime
    while (os.path.isdir(resultsFolder+ '_' + str(inc))):
        inc += 1
    resultsFolder = resultsFolder + '_' + str(inc)
    os.mkdir(resultsFolder)
    exp.setResultsFolder(resultsFolder)
    print("===> Saving the results of the experiment in {}".format(resultsFolder))

    # Creating directories for the trained models, the training and testing metrics
    # and the parameters of the model (i.e. the training parameters and the network
    # architecture)
    if (not doGridSearch):
        os.mkdir(resultsFolder + '/model/')
        os.mkdir(resultsFolder + '/metrics/')
    os.mkdir(resultsFolder + '/params_exp/')

    # Normalizing the dataset
    exp.compute_dataset_mean_std()
    exp.normalize_dataset()

    # Balancing the classes
    exp.balance_classes_loss()

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file = resultsFolder + '/params_exp/params_beginning' + '_'
    while (os.path.isfile(parameters_file + str(inc) + '.pth')):
        inc += 1
    parameters_file = parameters_file + str(inc) +'.pth'
    parameters_exp['audio_feature_shape'] = exp.audio_feature_shape
    with open(parameters_file, "wb") as fp:   #Pickling
        pickle.dump(parameters_exp, fp)

    # Evalauting the method
    if (not doGridSearch):
        # Doing holdout evaluation
        exp.holdout_train()
    else:
        # Doing grid search
        exp.gridSearch()

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file = resultsFolder + '/params_exp/params' + '_'
    while (os.path.isfile(parameters_file + str(inc) + '.pth')):
        inc += 1
    parameters_file = parameters_file + str(inc) +'.pth'
    parameters_exp['audio_feature_shape'] = exp.audio_feature_shape
    with open(parameters_file, "wb") as fp:   #Pickling
        pickle.dump(parameters_exp, fp)

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file = resultsFolder + '/params_exp/params' + '_'
    while (os.path.isfile(parameters_file + str(inc) + '.pth')):
        inc += 1
    parameters_file = parameters_file + str(inc) +'.pth'
    parameters_exp['audio_feature_shape'] = exp.audio_feature_shape
    with open(parameters_file, "wb") as fp:   #Pickling
        pickle.dump(parameters_exp, fp)

        # Saving the python file containing the network architecture
        if (parameters_exp['model_type'].lower() == '2dcnn'):
            if (parameters_exp['model_to_use'].lower() == 'timefrequency2dcnn'):
                shutil.copy2('../Models/CNNs/time_frequency_simple_CNN.py', resultsFolder + '/params_exp/network_architecture.py')
            elif (parameters_exp['model_to_use'].lower() == 'mnist2dcnn'):
                shutil.copy2('../Models/CNNs/mnist_CNN.py', resultsFolder + '/params_exp/network_architecture.py')
            else:
                raise ValueError('2D CNN {} is not valid'.format(parameters_exp['model_to_use']))

        elif (parameters_exp['model_type'].lower() == 'transformer'):
            if (parameters_exp['model_to_use'].lower() == 'rawaudiomultichannelcnn'):
                shutil.copy2('../Models/Transformers/Transformer_Encoder_RawAudioMultiChannelCNN.py', resultsFolder + '/params_exp/network_architecture.py')
            else:
                raise ValueError("Transformer type {} is not valid".format(parameters_exp['model_to_use']))
        else:
            raise ValueError("Model type {} is not valid".format(parameters_exp['model_type']))
    #==========================================================================#
    print("\n\n==================== End of the experiment ====================\n\n")



if __name__=="__main__":
    main()
