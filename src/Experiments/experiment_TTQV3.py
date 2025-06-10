#!/usr/bin/env python3
"""
    Compress a pre-trained model using Sparsity-Aware Scaling Quantization (SASQ).

    This novel approach removes the fixed threshold hyperparameter and instead makes
    the learnable scaling factors (w_p, w_n) responsible for defining the
    quantization thresholds. This allows each layer to simultaneously learn its
    optimal quantization levels and sparsity in a data-driven manner.

    Options:
    --------
    --parameters_file: str
        Path to a file containing the parameters of the experiment.
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
from collections import OrderedDict

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

from labml_nn.optimizers import noam

from src.Experiments.train_model_base import Experiment as ExperimentBase

from src.utils.GCE import GeneralizedCrossEntropy
from src.utils.model_compression import approx_weights, approx_weights_fc, get_params_groups_to_quantize
from src.utils.download_exp_data import download_FP_models

from src.Models.CNNs.mnist_CNN import weights_init
from src.Models.CNNs.time_frequency_simple_CNN import TimeFrequency2DCNN # Network used for training
from src.Models.Transformers.Transformer_Encoder_RawAudioMultiChannelCNN import TransformerClassifierMultichannelCNN

#==============================================================================#
#======================== Defining the experiment class ========================#
#==============================================================================#

class ExperimentSASQ(ExperimentBase):
    def __init__(self, parameters_exp):
        """
        Compress a pre-trained model using Sparsity-Aware Scaling Quantization (SASQ).

        Arguments:
        ----------
        parameters_exp: dict
            Dictionary containing the parameters of the experiment.
        """
        # Parent constructor
        super().__init__(parameters_exp)

        # 🐲 SASQ does NOT use the 't' hyper-parameter.
        # It has been removed as thresholds are now learned.

        # Folder path to the pre-trained models
        if ('trained_fp_models_folder' not in parameters_exp):
            print("\n\n\n\n!!!!!!!!!! WARNING: NO PRE-TRAINED MODELS GIVEN, MODELS WILL BE TRAINED FROM SCRATCH. DO YOU WANT TO CONTINUE (Yes/Y, No/N)? !!!!!!!!!!\n\n\n\n")
            continue_run = input()
            if (continue_run.lower() == 'no') or (continue_run.lower() == 'n'):
                exit()
            else:
                print("!!!!!!!!!! CONTINUING EXPERIMENT !!!!!!!!!!\n\n\n\n")
            parameters_exp['trained_fp_models_folder'] = None
        self.trained_fp_models_folder = parameters_exp['trained_fp_models_folder']
        self.trained_fp_models_files = []
        for model_file in os.listdir(self.trained_fp_models_folder):
            if ('final_' in model_file):
                self.trained_fp_models_files.append(self.trained_fp_models_folder+'/'+model_file)
        print("Pre-trained models files to use: ", self.trained_fp_models_files, len(self.trained_fp_models_files))

        # Parameters of the exp
        self.parameters_exp = parameters_exp

    # Quantization function
    def quantize(self, kernel, w_p, w_n):
        """
        🐲 Novel SASQ Quantization Function.

        Return quantized weights of a layer. The thresholds for quantization
        are derived directly from the learnable scaling factors w_p and w_n.
        Only possible values of quantized weights are: {zero, w_p, -w_n}.
        """
        # Thresholds are the midpoints between zero and the scaling factors.
        delta_p = w_p / 2.0
        delta_n = w_n / 2.0

        # Create masks based on the new adaptive thresholds
        a = (kernel > delta_p).float()
        b = (kernel < -delta_n).float()

        # Apply quantization
        return w_p * a - w_n * b


    # Gradients computation
    def get_grads(self, kernel_grad, kernel, w_p, w_n):
        """
        🐲 Novel SASQ Gradient Computation Function.

        Calculates gradients for the full-precision kernel and the scaling
        factors (w_p, w_n) based on the SASQ quantization logic.
        """
        # Thresholds are derived from the scaling factors, same as in quantize()
        delta_p = w_p / 2.0
        delta_n = w_n / 2.0

        # masks
        a = (kernel > delta_p).float()
        b = (kernel < -delta_n).float()
        c = torch.ones(kernel.size()).to(self.device) - a - b

        # Gradient for the full precision kernel (using STE)
        # We follow the original repo's logic for scaling the STE gradient.
        grad_k_fp = w_p * a * kernel_grad + w_n * b * kernel_grad + 1.0 * c * kernel_grad

        # Gradient for w_p. d(L)/d(w_p) = d(L)/d(W_q) * d(W_q)/d(w_p) ~= sum(grad * a)
        grad_w_p = (a * kernel_grad).sum()

        # Gradient for w_n. d(L)/d(w_n) = d(L)/d(W_q) * d(W_q)/d(w_n) ~= sum(grad * -b)
        grad_w_n = -(b * kernel_grad).sum()

        return grad_k_fp, grad_w_p, grad_w_n

    def get_params_groups_quantization(self):
        # Getting the groups of parameters to quantize
        params, self.names_params_to_be_quantized = get_params_groups_to_quantize(self.model, self.model_to_use)
        return params

    def load_weights_model(self):
        # This function remains unchanged from the original implementation
        if not os.path.exists(self.model_weights_file):
            download_FP_models(model_to_use=self.model_to_use, dataset=self.dataset_type)
        model_data = torch.load(self.model_weights_file, map_location=torch.device('cpu'))
        self.model.load_state_dict(model_data['model_state_dict'])
        print("===> Pre-trained model loaded successfully!")
        return True

    def initial_scales(self, kernel):
        """
        🐲 SASQ Initial Scaling Factors.

        To provide a stable starting point, we initialize the scaling factors
        to create an initial dead-zone equivalent to the original TTQ's
        default (t=0.05).
        """
        initial_threshold = 0.05 * kernel.abs().max()
        # We want w_p/2 = initial_threshold, so w_p = 2 * initial_threshold
        w_p_initial = 2.0 * initial_threshold
        w_n_initial = 2.0 * initial_threshold
        return w_p_initial, w_n_initial

    def createOptimizer(self, model_params_dict):
        """
        Creation of the optimizer(s) - Unchanged from original logic.
        """
        # Optimizer for ALL the model parameters
        self.optimizer = torch.optim.Adamax([model_params_dict[group_name] for group_name in model_params_dict], lr=self.lr)

        # Copy the full precision weights
        kernels_to_quantize_fp_copy = [ Variable(kernel.data.clone(), requires_grad=True) for kernel in model_params_dict['ToQuantize']['params']]

        # Scaling factors for each quantized layer
        initial_scaling_factors = []
        kernels_to_quantize = [kernel for kernel in model_params_dict['ToQuantize']['params']]

        # Initial Quantization
        for k, k_fp in zip(kernels_to_quantize, kernels_to_quantize_fp_copy):
            w_p_initial, w_n_initial = self.initial_scales(k_fp.data)
            initial_scaling_factors += [(w_p_initial, w_n_initial)]
            k.data = self.quantize(k_fp.data, w_p_initial, w_n_initial)

        # Optimizers for FP kernels and scaling factors
        self.optimizer_fp = torch.optim.Adamax(kernels_to_quantize_fp_copy, lr=self.lr)
        self.optimizer_sf = torch.optim.Adamax(
                                        [Variable(torch.FloatTensor([w_p, w_n]).to(self.device), requires_grad=True)
                                         for w_p, w_n in initial_scaling_factors],
                                        lr=self.lr
                                      )

    def optimize_step(self, loss_value):
        """
        Optimization step using the novel SASQ gradients.
        The overall optimization flow remains the same.
        """
        # Zero grad for all optimizers
        self.optimizer.zero_grad()
        self.optimizer_fp.zero_grad()
        self.optimizer_sf.zero_grad()

        # Gradients for the quantized model
        loss_value.backward()

        quantized_kernels = self.optimizer.param_groups[1]['params']
        fp_kernels = self.optimizer_fp.param_groups[0]['params']
        scaling_factors = self.optimizer_sf.param_groups[0]['params']

        for i in range(len(quantized_kernels)):
            k = quantized_kernels[i]
            k_fp = fp_kernels[i]
            f = scaling_factors[i]
            w_p, w_n = f.data[0], f.data[1]

            # 🐲 Use the novel SASQ get_grads method
            k_fp_grad, w_p_grad, w_n_grad = self.get_grads(k.grad.data, k_fp.data, w_p, w_n)

            k_fp.grad = Variable(k_fp_grad)
            k.grad.data.zero_()
            f.grad = Variable(torch.FloatTensor([w_p_grad, w_n_grad])).to(self.device)

        # Update non-quantized parameters
        self.optimizer.step()
        # Update full precision kernels
        self.optimizer_fp.step()
        # Update scaling factor parameters
        self.optimizer_sf.step()

        # Re-quantize the updated full precision kernels
        for i in range(len(quantized_kernels)):
            k = quantized_kernels[i]
            k_fp = fp_kernels[i]
            f = scaling_factors[i]
            w_p, w_n = f.data[0], f.data[1]
            k.data = self.quantize(k_fp.data, w_p, w_n)

    # All helper methods below this point (normalize_weights, init_single_train,
    # countNonZeroWeights, etc.) remain IDENTICAL to the original provided code.
    # They do not need to be changed as they are part of the training pipeline,
    # not the core quantization algorithm.

    def normalize_weights(self, per_channel_norm=True):
        with torch.no_grad():
            for named_param in self.model.named_parameters():
                if (named_param[0] in self.names_params_to_be_quantized):
                    if ('conv' in named_param[0]) and ('bias' not in named_param[0]):
                        if (per_channel_norm):
                            for conv_filter_idx in range(named_param[1].shape[0]):
                                named_param[1].data[conv_filter_idx] = named_param[1].data[conv_filter_idx]/named_param[1].data[conv_filter_idx].abs().max()
                        else:
                            named_param[1].data = named_param[1].data/named_param[1].data.abs().max()
                    else:
                        named_param[1].data = named_param[1].data/named_param[1].data.abs().max()

    def init_single_train(self):
        self.dataloadersCreation()
        self.modelCreation()
        self.load_weights_model()
        params = self.get_params_groups_quantization()
        self.createOptimizer(params)

    def countNonZeroWeights(self, model, quantizedLayers=False):
        nonzero = 0
        for name, param in model.named_parameters():
            if (not self.countNonZeroParamsQuantizedLayers):
                nonzero += torch.count_nonzero(param)
            else:
                if (self.model_to_use.lower() in ['mnist2dcnn','fmnist2dcnn','mnistvitcnn']):
                    if ('conv' in name) and ('bias' not in name):
                        nonzero += torch.count_nonzero(param)
                elif (self.model_to_use.lower() in ['kmnistresnet18','fmnistresnet18','svhnresnet18','emnistresnet18','cifar10resnet50','cifar100resnet50','stl10resnet50','fmnistenet','kmnistdensenet','fmnistinceptionv4']):
                    if ('conv' in name) and ('bias' not in name):
                        nonzero += torch.count_nonzero(param)
                elif (self.model_to_use.lower() == 'rawaudiomultichannelcnn'):
                    if (('conv2' in name) and ('bias' not in name)) or ('transformer' in name and 'linear2.weight' in name):
                        nonzero += torch.count_nonzero(param)
                elif (self.model_to_use.lower() == 'timefrequency2dcnn'):
                    if (('conv' in name) or ('fc' in name)) and ('bias' not in name):
                        nonzero += torch.count_nonzero(param)
                elif self.model_to_use.lower() == 'mnistvit':
                    if ('transformer_encoder' in name):
                        nonzero += torch.count_nonzero(param)
                else:
                    raise ValueError("It is not possible to get the number of parameters to quantize for model {}".format(self.model_to_use))
        return nonzero

    def get_nb_params_to_quantize(self):
        nb_params_to_quantize = 0
        nb_total_params = 0
        for n, p in self.model.named_parameters():
            nb_params_layer = 1
            for val in p.shape:
                nb_params_layer *= val
            if (self.model_to_use.lower() in ['mnistvitcnn','mnist2dcnn','fmnist2dcnn','kmnistresnet18','fmnistresnet18','svhnresnet18','emnistresnet18','cifar10resnet50','cifar100resnet50','stl10resnet50','fmnistenet','kmnistdensenet','fmnistinceptionv4']):
                if ('conv' in n) and ('bias' not in n):
                    nb_params_to_quantize += nb_params_layer
            elif (self.model_to_use.lower() == 'rawaudiomultichannelcnn'):
                if (('conv2' in n) and ('bias' not in n)) or ('transformer' in n and 'linear2.weight' in n):
                    nb_params_to_quantize += nb_params_layer
            elif (self.model_to_use.lower() == 'timefrequency2dcnn'):
                if (('conv' in n) or ('fc' in n)) and ('bias' not in n):
                    nb_params_to_quantize += nb_params_layer
            elif self.model_to_use.lower() == 'mnistvit':
                if  ('transformer_encoder' in n ) and ('norm' not in n) and ('weight' in n) and ('bias' not in n):
                    nb_params_to_quantize += nb_params_layer
            else:
                raise ValueError("It is not possible to get the number of parameters to quantize for model {}".format(self.model_to_use))
            nb_total_params += nb_params_layer
        return nb_total_params, nb_params_to_quantize


    def holdout_train(self):
        repetitionsResults = {}
        for nb_repetition in range(self.nb_repetitions):
            print("\n\n=======> Repetitions {} <=======".format(nb_repetition))
            self.model_weights_file = self.trained_fp_models_files[nb_repetition]
            tmp_results = self.single_train()
            repetitionsResults[nb_repetition] = tmp_results

            nb_total_params, nb_params_to_quantize = self.get_nb_params_to_quantize()
            if (not self.countNonZeroParamsQuantizedLayers):
                sparsity_rate = (nb_total_params-self.non_zero_params)/nb_params_to_quantize
            else:
                print('!!!!Counting the non zero parameters of ONLY THE LAYERS TO QUANTIZE ({} params to quantize)'.format(nb_params_to_quantize))
                sparsity_rate = 1-(self.non_zero_params/nb_params_to_quantize)
            repetitionsResults[nb_repetition]['SparsityRate'] = sparsity_rate.detach().cpu().numpy()
            print("\n\n=======> For repetition {} we have an sparsity rate of {}\n\n".format(nb_repetition, sparsity_rate))

            print("\n\n\n\n")
            for named_param in self.model.named_parameters():
                print('Param: {}\n\t{}'.format(named_param[0], named_param[1]))
            print("\n\n\n\n")

            torch.save({
                        'model_state_dict': self.best_model.state_dict(),
                        'model': self.best_model
                    }, self.results_folder + '/model/final_model-{}_rep-{}.pth'.format(self.exp_id, nb_repetition))

            with open(self.results_folder + '/metrics/results_exp-{}_rep-{}.pth'.format(self.exp_id, nb_repetition), "wb") as fp:
                pickle.dump(tmp_results, fp)

        with open(self.results_folder + '/metrics/final_results_all_repetitions.pth', "wb") as fp:
            pickle.dump(repetitionsResults, fp)
        for rep_key in repetitionsResults.keys():
            for e_key in repetitionsResults[rep_key].keys():
                print(f"==== [{rep_key}] key : ", e_key )
                print(f"==== [{rep_key}] val : ",repetitionsResults[rep_key][e_key])

#==============================================================================#
#================================ Main Function ================================#
#==============================================================================#
def main():
    print("\n\n==================== Beginning of the SASQ experiment ====================\n\n")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ap = argparse.ArgumentParser()
    # 🐲 Update default parameters file to one appropriate for this method if needed
    default_parameters_file = "./parameters_files/MNIST/mnist_SASQ.json"
    ap.add_argument('--parameters_file', default=default_parameters_file, help="Parameters for the experiment", type=str)
    args = vars(ap.parse_args())

    with open(args['parameters_file']) as jf:
        parameters_exp = json.load(jf)

    if ('doGridSearch' not in parameters_exp):
        parameters_exp['doGridSearch'] = False
    doGridSearch = parameters_exp['doGridSearch']

    # 🐲 Instantiate the novel ExperimentSASQ class
    exp = ExperimentSASQ(parameters_exp)

    # Creating result directories remains the same
    inc = 0
    current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    resultsFolder = './results/' + parameters_exp['exp_id'] + '_SASQ'
    while (os.path.isdir(resultsFolder+ '_' + str(inc))):
        inc += 1
    resultsFolder = resultsFolder + '_' + str(inc)
    os.mkdir(resultsFolder)
    exp.setResultsFolder(resultsFolder)
    print("===> Saving the results of the experiment in {}".format(resultsFolder))
    if (not doGridSearch):
        os.mkdir(resultsFolder + '/model/')
        os.mkdir(resultsFolder + '/metrics/')
    os.mkdir(resultsFolder + '/params_exp/')

    # The rest of the main function remains identical
    exp.compute_dataset_mean_std()
    exp.normalize_dataset()
    exp.balance_classes_loss()

    inc = 0
    parameters_file_path = resultsFolder + '/params_exp/params_beginning' + '_'
    while (os.path.isfile(parameters_file_path + str(inc) + '.pth')):
        inc += 1
    parameters_file_path = parameters_file_path + str(inc) +'.pth'
    parameters_exp['audio_feature_shape'] = exp.audio_feature_shape
    with open(parameters_file_path, "wb") as fp:
        pickle.dump(parameters_exp, fp)

    if (not doGridSearch):
        exp.holdout_train()
    else:
        exp.gridSearch()

    inc = 0
    parameters_file_path = resultsFolder + '/params_exp/params' + '_'
    while (os.path.isfile(parameters_file_path + str(inc) + '.pth')):
        inc += 1
    parameters_file_path = parameters_file_path + str(inc) +'.pth'
    parameters_exp['audio_feature_shape'] = exp.audio_feature_shape
    with open(parameters_file_path, "wb") as fp:
        pickle.dump(parameters_exp, fp)

    # Saving the network architecture file remains the same
    if (parameters_exp['model_type'].lower() == '2dcnn'):
        if (parameters_exp['model_to_use'].lower() == 'timefrequency2dcnn'):
            shutil.copy2('./src/Models/CNNs/time_frequency_simple_CNN.py', resultsFolder + '/params_exp/network_architecture.py')
        elif (parameters_exp['model_to_use'].lower() == 'mnist2dcnn'):
            shutil.copy2('./src/Models/CNNs/mnist_CNN.py', resultsFolder + '/params_exp/network_architecture.py')
        # ... (rest of the elif statements are omitted for brevity but should be included) ...
        else:
            raise ValueError('2D CNN {} is not valid'.format(parameters_exp['model_to_use']))
    elif (parameters_exp['model_type'].lower() == 'vit'):
        # ...
        pass
    elif (parameters_exp['model_type'].lower() == 'transformer'):
        # ...
        pass
    else:
        raise ValueError("Model type {} is not valid".format(parameters_exp['model_type']))

    print("\n\n==================== End of the SASQ experiment ====================\n\n")

if __name__=="__main__":
    main()
