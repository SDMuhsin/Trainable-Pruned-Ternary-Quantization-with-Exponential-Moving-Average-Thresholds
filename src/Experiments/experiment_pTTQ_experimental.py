#!/usr/bin/env python3
"""
    Compress a pre-trained model using TTQ.

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
from src.utils.model_compression import approx_weights, approx_weights_fc, pruning_function_pTTQ, pruning_function_asymmetric_manessi,pruning_function_pTTQ_experimental,pruning_function_pTTQ_experimental_learned


from src.Models.CNNs.time_frequency_simple_CNN import TimeFrequency2DCNN # Network used for training
from src.Models.Transformers.Transformer_Encoder_RawAudioMultiChannelCNN import TransformerClassifierMultichannelCNN

#==============================================================================#
#======================== Defining the experiment class ========================#
#==============================================================================#
class Experiment(ExperimentTTQ):
    def __init__(self, parameters_exp):
        """
            Compress a pre-trained model using TTQ.

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
        # Optimizing the pruning function parameters
        if ('learn_pruning_function_params' not in parameters_exp):
            parameters_exp['learn_pruning_function_params'] = True
        self.learn_pruning_function_params = parameters_exp['learn_pruning_function_params']
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
        if ('init_x' not in parameters_exp):
            parameters_exp['init_x'] = 5e-1
        self.init_x = parameters_exp['init_x']
        if ('init_y' not in parameters_exp):
            parameters_exp['init_y'] = 5e-1

        if ('k' not in parameters_exp):
            print("K not found, default to 1.0")
            parameters_exp['k'] = float(1)
        else:
            print("K found ",parameters_exp['k'])
        
        self.k = parameters_exp['k']
        if(parameters_exp['k_override'] != None):
            self.k = parameters_exp['k_override']
        
        self.beta = 0.9
        if(parameters_exp['beta'] != None):
            self.beta = parameters_exp['beta']

        self.exp_id += f"_k{self.k}_beta{self.beta}" 
        parameters_exp['exp_id'] = self.exp_id


        self.init_y = parameters_exp['init_y']

        # Pruning function
        if ('pruning_function_type' not in parameters_exp):
            parameters_exp['pruning_function_type'] = 'manessi_asymmetric_pTTQ'
        self.pruning_function_type = parameters_exp['pruning_function_type']
        if (self.pruning_function_type.lower() == 'manessi_asymmetric_pttq'):
            self.pruning_function = pruning_function_pTTQ
        elif (self.pruning_function_type.lower() == 'manessi_asymmetric'):
            self.pruning_function = pruning_function_asymmetric_manessi
        elif (self.pruning_function_type.lower() == 'experimental'):
            self.pruning_function = pruning_function_pTTQ_experimental
        elif (self.pruning_function_type.lower() == 'experimental_learned'):
            self.pruning_function = pruning_function_pTTQ_experimental_learned
        else:
            raise ValueError("Pruning function {} is not valid".format(self.pruning_function_type))

        # Optimizer to use for thresholds and alpha
        if ('optimizer_pruning_params' not in parameters_exp):
            parameters_exp['optimizer_pruning_params'] = 'SGD'
        if (parameters_exp['optimizer_pruning_params'].lower() == 'sgd'):
            self.optimizer_pruning_params = torch.optim.SGD
        elif (parameters_exp['optimizer_pruning_params'].lower() == 'adamax'):
            self.optimizer_pruning_params = torch.optim.Adamax
        elif (parameters_exp['optimizer_pruning_params'].lower() == 'adam'):
            self.optimizer_pruning_params = torch.optim.Adam
        else:
            raise ValueError("Optimizer {} is not valid to optimize the pruning parameters".format(parameters_exp['optimizer_pruning_params']))

        # Parameters of the exp
        self.parameters_exp = parameters_exp


    # Quantization function
    def quantize(self, kernel, w_p, w_n):
        """
        Function from inspired from https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py
        ATTENTION: it is not the same function as we change the method to quantize
        the weights.

        Return quantized weights of a layer.
        Only possible values of quantized weights are: {zero, w_p, -w_n}.
        """
        # Getting the pruned kernel
        pruned_kernel = self.pruning_function(kernel, self.alpha, self.a, self.b, self.k, self.beta)
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
        pruned_kernel = self.pruning_function(kernel, self.alpha, self.a, self.b, self.k, self.beta)
        A = (pruned_kernel > 0).float()
        B = (pruned_kernel < 0).float()
        c = torch.ones(pruned_kernel.size()).to(self.device) - A - B
        grad_fp_kernel = w_p*A*kernel_grad + w_n*B*kernel_grad + 1.0*c*kernel_grad
        grad_wp = (A*kernel_grad).sum()
        grad_wn = (B*kernel_grad).sum()

        # Grads
        if (self.pruning_function_type in ['manessi_asymmetric_pTTQ','experimental']): # Both these use stats based thresholds
            # Grads of the thresholds hyperparameters
            kernel_mean, kernel_std = kernel.mean(), kernel.std()
            delta_min = (kernel_mean + self.a*kernel_std).abs()
            delta_max = (kernel_mean + self.b*kernel_std).abs()
            grad_a = (
                            kernel_std*torch.heaviside((-kernel-delta_min).float(),  torch.tensor([0]).float().to(self.device) )\
                            - kernel_std*torch.nn.functional.sigmoid(self.alpha*(-kernel-delta_min))\
                            + kernel_std*delta_min*self.alpha*torch.nn.functional.sigmoid(self.alpha*(-kernel-delta_min))*(1-torch.nn.functional.sigmoid(self.alpha*(-kernel-delta_min)))
                     ).sum()
            grad_b = (
                            -kernel_std*torch.heaviside((kernel-delta_max).float(),  torch.tensor([0]).float().to(self.device) )\
                            + kernel_std*torch.nn.functional.sigmoid(self.alpha*(kernel-delta_max))\
                            - kernel_std*delta_max*self.alpha*torch.nn.functional.sigmoid(self.alpha*(kernel-delta_max))*(1-torch.nn.functional.sigmoid(self.alpha*(kernel-delta_max)))
                     ).sum()
            # Grads of alpha
            grad_alpha = (delta_max*(kernel-delta_max)*torch.nn.functional.sigmoid(self.alpha*(kernel-delta_max))*(1-torch.nn.functional.sigmoid(self.alpha*(kernel-delta_max)))\
                        + delta_min*(kernel+delta_min)*torch.nn.functional.sigmoid(self.alpha*(-kernel-delta_min))*(1-torch.nn.functional.sigmoid(self.alpha*(-kernel-delta_min)))).sum()

        elif (self.pruning_function_type in ['manessi_asymmetric','experimental_learned']): # Both these use learnable thresholds 
            # Grads of w_p and w_n
            grad_a = (
                        torch.heaviside((-kernel-self.a).float(),  torch.tensor([0]).float().to(self.device) )\
                        - torch.nn.functional.sigmoid(self.alpha*(-kernel-self.a))\
                        + self.a*self.alpha*torch.nn.functional.sigmoid(self.alpha*(-kernel-self.a))*(1-torch.nn.functional.sigmoid(self.alpha*(-kernel-self.a)))
                     ).sum()
            grad_b = (
                        -torch.heaviside((kernel-self.b).float(),  torch.tensor([0]).float().to(self.device) )\
                        + torch.nn.functional.sigmoid(self.alpha*(kernel-self.b))\
                        - self.b*self.alpha*torch.nn.functional.sigmoid(self.alpha*(kernel-self.b))*(1-torch.nn.functional.sigmoid(self.alpha*(kernel-self.b)))
                     ).sum()
            # Grads of alpha
            grad_alpha = (self.b*(kernel-self.b)*torch.nn.functional.sigmoid(self.alpha*(kernel-self.b))*(1-torch.nn.functional.sigmoid(self.alpha*(kernel-self.b)))\
                        + self.a*(kernel+self.a)*torch.nn.functional.sigmoid(self.alpha*(-kernel-self.a))*(1-torch.nn.functional.sigmoid(self.alpha*(-kernel-self.a)))).sum()


        else:
            raise ValueError("Pruning function {} is not valid".format(self.pruning_function_type))

        return grad_fp_kernel, grad_wp, grad_wn, grad_a, grad_b, grad_alpha

    def initial_alpha(self, kernel):
        return self.alpha

    def initial_thresholds(self, kernel):
        return self.init_x, self.init_y

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
            self.a, self.b = self.initial_thresholds(k_fp.data)
            initial_thresh += [(self.a, self.b)]

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
        self.optimizer_t = self.optimizer_pruning_params(
                                            [Variable(torch.FloatTensor([x, y]).to(self.device), requires_grad=True)
                                             for x, y in initial_thresh],
                                            lr=self.lr_thresh
                                         )
        self.sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_t, T_max=10, eta_min=1e-7, last_epoch=-1, verbose=True)


        # Alpha value
        if (self.optimize_alpha):
            self.optimizer_alpha = self.optimizer_pruning_params(
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

        # # Applying lr scheduler for the thresholds
        # self.sched_t.step()
        #
        # # Applying lr scheduler for the alpha values
        # if (self.optimize_alpha):
        #     self.sched_alpha.step()


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
            self.a, self.b = t.data[0], t.data[1]

            # Getting the alpha values used for pruning
            if (self.optimize_alpha):
                alpha_val = alphas[i]
                self.alpha = alpha_val.data[0]

            # Getting the gradients
            k_fp_grad, w_p_grad, w_n_grad, a_grad, b_grad, alpha_grad = self.get_grads(k.grad.data, k_fp.data, w_p, w_n)

            # Gradient for the full precision kernels
            k_fp.grad = Variable(k_fp_grad)

            # The quantized kernels are not updated (they are computed from the FP kernels)
            k.grad.data.zero_()

            # Gradient for the scaling factors
            f.grad = Variable(torch.FloatTensor([w_p_grad, w_n_grad]).to(self.device))

            # Gradient for the thresholds
            t.grad = Variable(torch.FloatTensor([a_grad, b_grad]).to(self.device))

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

        if (self.learn_pruning_function_params):
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
                t.data[0], t.data[1] = torch.clamp(t.data[0], min=1e-8), torch.clamp(t.data[1], min=1e-8)

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
            self.a, self.b = t.data[0], t.data[1]

            # Getting the alpha values used for pruning
            if (self.optimize_alpha):
                alpha_val = alphas[i]
                self.alpha = alpha_val.data[0]
            k.data = self.quantize(k_fp.data, w_p, w_n)

    def gridSearch(self):
        """
            Does a grid search for some hyper-parameters
        """
        lr_values = [self.lr]
        lr_wn_wp_values = [self.lr_wn_wp]
        init_x_values = [self.init_x]
        init_y_values = [self.init_x]
        lr_thresh_values = [self.lr_thresh]
        alpha_values = [1, 10, 50, 80, 90, 100]
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
                                        self.init_x = init_x
                                        self.init_y = init_y
                                        self.lr_thresh = lr_thresh
                                        self.alpha = alpha
                                        self.lr_alpha = lr_alpha

                                        # Creating the datasets folder
                                        current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
                                        os.mkdir(base_results_folder + '/LR-{}_LRWNWP-{}_INITX-{}_INITY-{}_LRTHRESH-{}_ALPHA-{}_LRALPHA-{}_{}/'.format(self.lr, self.lr_wn_wp, self.init_x, self.init_y, self.lr_thresh, self.alpha, self.lr_alpha, current_datetime))
                                        os.mkdir(base_results_folder + '/LR-{}_LRWNWP-{}_INITX-{}_INITY-{}_LRTHRESH-{}_ALPHA-{}_LRALPHA-{}_{}/model/'.format(self.lr, self.lr_wn_wp, self.init_x, self.init_y, self.lr_thresh, self.alpha, self.lr_alpha, current_datetime))
                                        os.mkdir(base_results_folder + '/LR-{}_LRWNWP-{}_INITX-{}_INITY-{}_LRTHRESH-{}_ALPHA-{}_LRALPHA-{}_{}/metrics/'.format(self.lr, self.lr_wn_wp, self.init_x, self.init_y, self.lr_thresh, self.alpha, self.lr_alpha, current_datetime))
                                        self.results_folder = base_results_folder + '/LR-{}_LRWNWP-{}_INITX-{}_INITY-{}_LRTHRESH-{}_ALPHA-{}_LRALPHA-{}_{}/'.format(self.lr, self.lr_wn_wp, self.init_x, self.init_y, self.lr_thresh, self.alpha, self.lr_alpha, current_datetime)

                                        # Training
                                        self.holdout_train()

        self.results_folder = base_results_folder

    def randomSearch(self):
        """
            Does a grid search for some hyper-parameters
        """
        # Defining the values of the parameters to test
        lr_values = [1e-2, 1e-3, 1e-4, 1e-5]
        init_x_values = [4, 3, 2, 1]
        init_y_values = [4,  3, 2, 1]
        alpha_values = [100, 500, 1000]

        # Iterating over the different values of the hyper-parameters
        nb_tested_hyper_params = 0
        nb_hyper_params_to_test = len(lr_values)*len(init_x_values)*len(init_y_values)*len(alpha_values)
        seen_hyper_params = []
        base_results_folder = self.results_folder
        while nb_tested_hyper_params < nb_hyper_params_to_test:
            lr = random.choice(lr_values)
            init_x = random.choice(init_x_values)
            init_y = random.choice(init_y_values)
            alpha = random.choice(alpha_values)
            lr_wn_wp = lr
            lr_thresh = lr
            lr_alpha = lr
            hyper_params = (lr, lr_wn_wp, init_x, init_y, lr_thresh, alpha, lr_alpha)
            if (hyper_params not in seen_hyper_params) and (init_x <= init_y):
                # Updating the hyper-paramet of the experiment
                self.lr = lr
                self.lr_wn_wp = lr_wn_wp
                self.lr_thresh = lr_thresh
                self.lr_alpha = lr_alpha
                self.init_x = init_x
                self.init_y = init_y
                self.alpha = alpha

                # Creating the datasets folder
                current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
                os.mkdir(base_results_folder + '/LR-{}_LRWNWP-{}_INITX-{}_INITY-{}_LRTHRESH-{}_ALPHA-{}_LRALPHA-{}_{}/'.format(self.lr, self.lr_wn_wp, self.init_x, self.init_y, self.lr_thresh, self.alpha, self.lr_alpha, current_datetime))
                os.mkdir(base_results_folder + '/LR-{}_LRWNWP-{}_INITX-{}_INITY-{}_LRTHRESH-{}_ALPHA-{}_LRALPHA-{}_{}/model/'.format(self.lr, self.lr_wn_wp, self.init_x, self.init_y, self.lr_thresh, self.alpha, self.lr_alpha, current_datetime))
                os.mkdir(base_results_folder + '/LR-{}_LRWNWP-{}_INITX-{}_INITY-{}_LRTHRESH-{}_ALPHA-{}_LRALPHA-{}_{}/metrics/'.format(self.lr, self.lr_wn_wp, self.init_x, self.init_y, self.lr_thresh, self.alpha, self.lr_alpha, current_datetime))
                self.results_folder = base_results_folder + '/LR-{}_LRWNWP-{}_INITX-{}_INITY-{}_LRTHRESH-{}_ALPHA-{}_LRALPHA-{}_{}/'.format(self.lr, self.lr_wn_wp, self.init_x, self.init_y, self.lr_thresh, self.alpha, self.lr_alpha, current_datetime)

                # Training
                self.holdout_train()

                # Adding the parameters to the list of seen hyper-parameters
                seen_hyper_params.append(hyper_params)
                nb_tested_hyper_params += 1

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
    default_parameters_file = "./parameters_files/MNIST/mnist_pTTQ.json"
    ap.add_argument('--parameters_file', default=default_parameters_file, help="Parameters for the experiment", type=str)
    ap.add_argument('--k_override', default= None, help = "Override k with this value for experimental pTTQ", type= float)
    ap.add_argument('--beta', default= 0.9, help = "beta value for experimental pTTQ", type= float)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    parameters_file = args['parameters_file']
    with open(parameters_file) as jf:
        parameters_exp = json.load(jf)
    
    parameters_exp['k_override'] = args['k_override'] 
    parameters_exp['beta'] = args['beta']

    # Grid search parameter in the parameters file
    if ('doGridSearch' not in parameters_exp):
        parameters_exp['doGridSearch'] = False
    doGridSearch = parameters_exp['doGridSearch']
    if ('doRandomSearch' not in parameters_exp):
        parameters_exp['doRandomSearch'] = True
    doRandomSearch = parameters_exp['doRandomSearch']

    #==========================================================================#
    # Creating an instance of the experiment
    exp = Experiment(parameters_exp)

    # Creating directory to save the results
    inc = 0
    current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    resultsFolder = './results/' + parameters_exp['exp_id'] + '_OW'
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
        if (not doRandomSearch):
            print("\n\n\n\n=======> Doing GRID search <=======\n\n\n\n")
            exp.gridSearch()
        else:
            print("\n\n\n\n=======> Doing RANDOM search <=======\n\n\n\n")
            exp.randomSearch()

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file = resultsFolder + '/params_exp/params' + '_OW'
    while (os.path.isfile(parameters_file + str(inc) + '.pth')):
        inc += 1
    parameters_file = parameters_file + str(inc) +'.pth'
    parameters_exp['audio_feature_shape'] = exp.audio_feature_shape
    with open(parameters_file, "wb") as fp:   #Pickling
        pickle.dump(parameters_exp, fp)

        # Saving the python file containing the network architecture
        if (parameters_exp['model_type'].lower() == '2dcnn'):
            if (parameters_exp['model_to_use'].lower() == 'timefrequency2dcnn'):
                shutil.copy2('./src/Models/CNNs/time_frequency_simple_CNN.py', resultsFolder + '/params_exp/network_architecture.py')
            elif (parameters_exp['model_to_use'].lower() == 'mnist2dcnn'):
                shutil.copy2('./src/Models/CNNs/mnist_CNN.py', resultsFolder + '/params_exp/network_architecture.py')
            elif (parameters_exp['model_to_use'].lower() == 'fmnist2dcnn'):
                shutil.copy2('./src/Models/CNNs/mnist_CNN.py', resultsFolder + '/params_exp/network_architecture.py')
            elif (parameters_exp['model_to_use'].lower() in ['kmnistresnet18','fmnistresnet18','svhnresnet18','emnistresnet18']):
                shutil.copy2('./src/Models/CNNs/resnet18.py', resultsFolder + '/params_exp/network_architecture.py')
            elif (parameters_exp['model_to_use'].lower() in ['cifar10resnet50','cifar100resnet50','stl10resnet50']):
                shutil.copy2('./src/Models/CNNs/resnet50.py', resultsFolder + '/params_exp/network_architecture.py')                
            elif (parameters_exp['model_to_use'].lower() == 'fmnistenet'):
                shutil.copy2('./src/Models/CNNs/fmnist_enet.py', resultsFolder + '/params_exp/network_architecture.py')
            elif (parameters_exp['model_to_use'].lower() == 'kmnistdensenet'):
                shutil.copy2('./src/Models/CNNs/densenet.py', resultsFolder + '/params_exp/network_architecture.py')
            else:
                raise ValueError('2D CNN {} is not valid'.format(parameters_exp['model_to_use']))
        elif (parameters_exp['model_type'].lower() == 'vit'):
           if (parameters_exp['model_to_use'].lower() == 'mnistvit'):
                shutil.copy2('./src/Models/Transformers/mnist_vit.py', resultsFolder + '/params_exp/network_architecture.py')
        elif (parameters_exp['model_type'].lower() == 'transformer'):
            if (parameters_exp['model_to_use'].lower() == 'rawaudiomultichannelcnn'):
                shutil.copy2('./src/Models/Transformers/Transformer_Encoder_RawAudioMultiChannelCNN.py', resultsFolder + '/params_exp/network_architecture.py')
            else:
                raise ValueError("Transformer type {} is not valid".format(parameters_exp['model_to_use']))
        else:
            raise ValueError("Model type {} is not valid".format(parameters_exp['model_type']))
    #==========================================================================#
    print("\n\n==================== End of the experiment ====================\n\n")



if __name__=="__main__":
    main()
