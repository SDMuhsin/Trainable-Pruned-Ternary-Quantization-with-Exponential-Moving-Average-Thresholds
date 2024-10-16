#!/usr/bin/env python3
"""
    Compress a pre-trained model using a binary DoReFaNet.

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

# class Experiment(ExperimentOneFeature):
class Experiment(ExperimentBase):
    def __init__(self, parameters_exp):
        """
            Compress a pre-trained model using a DoReFaNet Binary method.

            Arguments:
            ----------
            parameters_exp: dict
                Dictionary containing the parameters of the experiment:
                    * exp_id: str, name of the experiment.
                    * feature_type
        """
        # Parent constructor
        super().__init__(parameters_exp)

        # Folder path to the pre-trained models
        if ('trained_fp_models_folder' not in parameters_exp):
            print("\n\n\n\n!!!!!!!!!! WARNING: NO PRE-TRAINED MODELS GIVEN, MODELS WILL BE TRAINED FROM SCRATCH. DO YOU WANT TO CONTINUE (Yes/Y, No/N)? !!!!!!!!!!\n\n\n\n")
            continue_run = input()
            if (continue_run.lower() == 'no') or (continue_run.lower() == 'n'):
                exit()
            # elif (continue_run.lower() == 'yes') or (continue_run.lower() == 'y'):
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
    def quantize(self, kernel):
        """
            Quantization done according to the DoReFaNet and TTQ papers.
        """
        w_b = kernel.abs().mean()*( (kernel >= 0.).float() - (kernel < 0.).float() )

        return w_b


    # Gradients computation
    def get_grads(self, kernel_grad):
        """
        Grads computed according to the DoReFaNet and TTQ papers.

        Arguments:
        ----------
            kernel_grad: gradient with respect to quantized kernel.

        Returns:
        --------
            1. gradient for the full precision kernel.
        """
        return kernel_grad


    def get_params_groups_quantization(self):
        # Getting the groups of parameters to quantize
        params, self.names_params_to_be_quantized = get_params_groups_to_quantize(self.model, self.model_to_use)
        return params

    def load_weights_model(self):
        # Verifying if the model's weights file exists
        if not (os.path.exists(self.model_weights_file)):
            download_FP_models(model_to_use=self.model_to_use, dataset=self.dataset_type)

        # Loading the data of a model
        model_data = torch.load(self.model_weights_file, map_location=torch.device('cpu'))

        # Loading the weights into the model
        self.model.load_state_dict(model_data['model_state_dict'])
        print("===> Model loaded successfully !")

    def createOptimizer(self, model_params_dict):
        """
            Creation of the optimizer(s)
        """
        # Optimizer for ALL the model parameters
        self.optimizer = torch.optim.Adamax([model_params_dict[group_name] for group_name in model_params_dict], lr=self.lr)

        # Copy the full precision weights of the parameters that are going to be
        # quantized (so not all the FP weights)
        kernels_to_quantize_fp_copy = [ Variable(kernel.data.clone(), requires_grad=True) for kernel in model_params_dict['ToQuantize']['params']]

        # Scaling factors for each quantized layer
        initial_scaling_factors = []

        # Kernels to be quantized
        kernels_to_quantize = [kernel for kernel in model_params_dict['ToQuantize']['params']]

        # Initial Quantization
        for k, k_fp in zip(kernels_to_quantize, kernels_to_quantize_fp_copy):
            # Doing quantization
            k.data = self.quantize(k_fp.data)

        # Getting the optimizers for the FP kernels and the scaling factors
        # FP kernels
        self.optimizer_fp = torch.optim.Adamax(kernels_to_quantize_fp_copy, lr=self.lr)

    def optimize_step(self, loss_value):
        """
            Simple optimization step
        """
        # Zero grad for all optimizers
        self.optimizer.zero_grad()
        self.optimizer_fp.zero_grad()

        # Gradients for the quantized model
        loss_value.backward()

        # Gett the quantized kernels
        quantized_kernels = self.optimizer.param_groups[1]['params']

        # Get the FP copies of the original kernels
        fp_kernels = self.optimizer_fp.param_groups[0]['params']

        for i in range(len(quantized_kernels)):
            # Current quantized kernel
            k = quantized_kernels[i]

            # Getting the FP version of the current kernel
            k_fp = fp_kernels[i]

            # Getting the gradients
            k_fp_grad = self.get_grads(k.grad.data)

            # Gradient for the full precision kernels
            k_fp.grad = Variable(k_fp_grad)

            # The quantized kernels are not updated (they are computed from the FP kernels)
            k.grad.data.zero_()

        # Update all the parameters that should not be quantized (usually the first and last
        # layers, as well as the batch norm parameters)
        self.optimizer.step()

        # Update the full precision kernels
        self.optimizer_fp.step()

        # Quantize the updated full precision kernels
        for i in range(len(quantized_kernels)):
            k = quantized_kernels[i]
            k_fp = fp_kernels[i]
            k.data = self.quantize(k_fp.data)

    def normalize_weights(self, per_channel_norm=True):
        """
            Normalize the weights of a model.
            Convolutions are normalized PER CHANNEL.
            The rest of the layers are normalized at a layer level.

            Arguments:
            ----------
            model: torch model
                Model from which we want to normalize the weights.
            per_channel_norm: bool
                Bool indicating if normalization is done per channel for convolutional
                layers
        """
        # print("\n\n\n\nNormalizing ONLY the WEIGHTS to be QUANTIZED !!!!!!\n\n\n\n")
        with torch.no_grad():
            for named_param in self.model.named_parameters():
                if (named_param[0] in self.names_params_to_be_quantized):
                    # print("\n======>Normalizing param {} <=======\n\n".format(named_param[0]))
                    #print('\n\nParam: {}\n'.format(named_param[0]))
                    # Doing it layer by layer and channel by channel
                    if ('conv' in named_param[0]) and ('bias' not in named_param[0]):
                        if (per_channel_norm):
                            for conv_filter_idx in range(named_param[1].shape[0]):
                                named_param[1].data[conv_filter_idx] = named_param[1].data[conv_filter_idx]/named_param[1].data[conv_filter_idx].abs().max()
                        else:
                            named_param[1].data = named_param[1].data/named_param[1].data.abs().max()
                    else:
                        named_param[1].data = named_param[1].data/named_param[1].data.abs().max()


    def init_single_train(self):
        """
            Initialize the dataloaders, models and optimizers for a single train
        """
        #======================================================================#
        #================ Initialization Model and data loader ================#
        #======================================================================#
        # Creating the dataloaders
        self.dataloadersCreation()

        # Creating the model
        self.modelCreation()

        # Initializing weights
        # self.model.apply(weights_init)

        # Loading the weights of the trained models
        self.load_weights_model()

        # # Normalizing the weights
        # if (self.do_normalization_weights):
        #     self.normalize_weights(per_channel_norm=True)
        #     # self.normalize_weights(per_channel_norm=False)


        #======================================================================#
        #========= Initialization Quantization Params and optimizers =========#
        #======================================================================#
        # Group of parameters
        params = self.get_params_groups_quantization()

        # Optimizer for the model parameters
        # self.createOptimizer(params)
        self.createOptimizer(params)

    def countNonZeroWeights(self, model, quantizedLayers=False):
        """
            Count the number of non zero parameters in the model

            Arguments:
            ----------
            model: torch model
                Model from which we want to count the weights.
            quantizedLayers: boolean
                True if want to count the non zero weights of the weights to
                quantize.
        """
        nonzero = 0
        for name, param in model.named_parameters():
            if (not self.countNonZeroParamsQuantizedLayers):
                nonzero += torch.count_nonzero(param)
            else:
                # More general method
                if (self.model_to_use.lower() == 'mnist2dcnn')\
                    or (self.model_to_use.lower() == 'rawaudiomultichannelcnn')\
                    or (self.model_to_use.lower() == 'timefrequency2dcnn'):
                        if (name in self.names_params_to_be_quantized):
                             nonzero += torch.count_nonzero(param)

                else:
                    raise ValueError("It is not possible to get the number of parameters to quantize for model {}".format(self.model_to_use))


        return nonzero

    def get_nb_params_to_quantize(self):
        nb_params_to_quantize = 0
        nb_total_params = 0
        for n, p in self.model.named_parameters():
            # Nb params layer
            nb_params_layer = 1
            for val in p.shape:
                nb_params_layer *= val

            # Nb params quantize
            if (self.model_to_use.lower() == 'mnist2dcnn')\
                or (self.model_to_use.lower() == 'rawaudiomultichannelcnn')\
                or (self.model_to_use.lower() == 'timefrequency2dcnn'):
                    if (n in self.names_params_to_be_quantized):
                            nb_params_to_quantize += nb_params_layer
            else:
                raise ValueError("It is not possible to get the number of parameters to quantize for model {}".format(self.model_to_use))

            # Nb tot params
            nb_total_params += nb_params_layer

        return nb_total_params, nb_params_to_quantize


    def holdout_train(self):
        """
            Does a holdout training repeated self.nb_repetitions times
        """
        repetitionsResults = {}
        for nb_repetition in range(self.nb_repetitions):
            print("\n\n=======> Repetitions {} <=======".format(nb_repetition))
            # Get pre-trained model
            self.model_weights_file = self.trained_fp_models_files[nb_repetition]

            # Doing single train
            tmp_results = self.single_train()
            repetitionsResults[nb_repetition] = tmp_results


            # Sparsity rate
            nb_total_params, nb_params_to_quantize = self.get_nb_params_to_quantize()
            if (not self.countNonZeroParamsQuantizedLayers):
                sparsity_rate = (nb_total_params-self.non_zero_params)/nb_params_to_quantize
            else:
                print('!!!!Counting the non zero parameters of ONLY THE LAYERS TO QUANTIZE ({} params to quantize over {})'.format(nb_params_to_quantize, nb_total_params))
                sparsity_rate = 1-(self.non_zero_params/nb_params_to_quantize)
            repetitionsResults[nb_repetition]['SparsityRate'] = sparsity_rate.detach().cpu().numpy()
            print("\n\n=======> For repetition {} we have an sparsity rate of {}\n\n".format(nb_repetition, sparsity_rate))

            # Parameters of the model
            print("\n\n\n\n")
            for named_param in self.model.named_parameters():
                print('Param: {}\n\t{}'.format(named_param[0], named_param[1]))
            print("\n\n\n\n")

            # Saving the final model and the results
            # Model
            torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'model': self.model
                        }, self.results_folder + '/model/final_model-{}_rep-{}.pth'.format(self.exp_id, nb_repetition))
            # # Model exported with torch.jit
            # model_jit = torch.jit.script(self.model) # Export to TorchScript
            # model_jit.save(self.results_folder + '/model/final-JIT_model-{}_rep-{}.pth'.format(self.exp_id, nb_repetition)) # Save
            # Results
            with open(self.results_folder + '/metrics/results_exp-{}_rep-{}.pth'.format(self.exp_id, nb_repetition), "wb") as fp:   #Pickling
                pickle.dump(tmp_results, fp)

        # Saving the results of the different repetitions
        with open(self.results_folder + '/metrics/final_results_all_repetitions.pth', "wb") as fp:   #Pickling
            pickle.dump(repetitionsResults, fp)


#==============================================================================#
#================================ Main Function ================================#
#==============================================================================#
def main():
    print("\n\n==================== Beginning of the experiment ====================\n\n")
    #==========================================================================#
    # Fixing the random seed
    seed = 1
    # seed = 42
    random.seed(seed) # For reproducibility purposes
    np.random.seed(seed) # For reproducibility purposes
    torch.manual_seed(seed) # For reproducibility purposes
    if torch.cuda.is_available(): # For reproducibility purposes
        torch.cuda.manual_seed_all(seed)

    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    default_parameters_file = "../../../parameters_files/model_compression/spectrogram.json"
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
