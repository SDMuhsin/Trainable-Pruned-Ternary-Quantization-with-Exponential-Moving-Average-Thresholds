#!/usr/bin/env python3
"""
    Trains a model using a given dataset.

    Options:
    --------
    --parameters_file: str
        Path to a file containing the parameters of the experiment.
        This files are usually located in /hits_signal_learning/parameters_files/
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
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from labml_nn.optimizers import noam

from src.utils.GCE import GeneralizedCrossEntropy
from src.utils.raw_audio_tools import min_max_audio_lenght
from src.utils.download_exp_data import download_ESR
from src.utils.tools import change_data_samples_path, string_to_bool, train_val_split_stratified, balance_dataset

from src.DataManipulation.mnist_data import put_MNIST_data_generic_form, MNISTDatasetWrapper
from src.DataManipulation.eeg_data import EEG_EpilepticSeizureRecognition, loadFromHDF5_EEG
from src.DataManipulation.svhn_data import put_SVHN_data_generic_form, SVHNDatasetWrapper
from src.DataManipulation.emnist_data import put_EMNIST_data_generic_form. EMNISTDatasetWrapper

from src.Models.CNNs.mnist_CNN import MnistClassificationModel, weights_init 
from src.Models.CNNs.resnet18 import ResNet18ClassificationModel
from src.Models.CNNs.time_frequency_simple_CNN import TimeFrequency2DCNN
from src.Models.Transformers.Transformer_Encoder_RawAudioMultiChannelCNN import TransformerClassifierMultichannelCNN

#==============================================================================#
#======================== Defining the experiment class ========================#
#==============================================================================#

class Experiment(object):
    def __init__(self, parameters_exp):
        """
            Trains a model using a given dataset.

            Arguments:
            ----------
            parameters_exp: dict
                Dictionary containing the parameters of the experiment:
                    * exp_id: str, name of the experiment.
                    * feature_type
        """
        # Defining some attributes of the experiment
        self.exp_id = parameters_exp['exp_id']
        self.results_folder = None

        # Feature type
        if ('feature_type' not in parameters_exp):
            parameters_exp['feature_type'] = 'RawSignal'
        self.feature_type = parameters_exp['feature_type']

        # Dataset type
        if ('dataset_type' not in parameters_exp):
            parameters_exp['dataset_type'] = 'ESR'
        self.dataset_type = parameters_exp['dataset_type']

        # Warmup steps (for NOAM when it is used)
        if ('warmup' not in parameters_exp):
            parameters_exp['warmup'] = 4000
        self.warmup = parameters_exp['warmup']

        # Optimize hyperparams
        if ('do_hyperparams_opt' not in parameters_exp):
            parameters_exp['do_hyperparams_opt'] = False
        self.do_hyperparams_opt = parameters_exp['do_hyperparams_opt']

        # Count only the non zero params of the layers to quantize
        if ('countNonZeroParamsQuantizedLayers' not in parameters_exp):
            parameters_exp['countNonZeroParamsQuantizedLayers'] = True
        self.countNonZeroParamsQuantizedLayers = parameters_exp['countNonZeroParamsQuantizedLayers']

        # Model type to use (2D CNN, 1D CNN-Transformer)
        if ('model_type' not in parameters_exp):
            parameters_exp['model_type'] = 'Transformer'
        self.model_type = parameters_exp['model_type']

        # Precise model to use
        if ('model_to_use' not in parameters_exp):
            if (parameters_exp['model_type'].lower() == '2dcnn'):
                parameters_exp['model_to_use'] = 'TimeFrequency2DCNN'
            elif (parameters_exp['model_type'].lower() == 'transformer'):
                parameters_exp['model_to_use'] = 'RawAudioMultiChannelCNN'
            else:
                raise ValueError("Model type {} is not valid".format(parameters_exp['model_type']))
        self.model_to_use = parameters_exp['model_to_use']

        # Use class token parameter if the model type is a Transformer
        if (parameters_exp['model_type'].lower() == 'transformer'):
            if ('classification_pool' not in parameters_exp):
                parameters_exp['classification_pool'] = 'ClassToken'

        # Positional encoding to use if a Transformer model is used
        if (parameters_exp['model_type'].lower() == 'transformer'):
            if ('pos_encoder_type' not in parameters_exp):
                parameters_exp['pos_encoder_type'] = 'Original'
            self.pos_encoder_type = parameters_exp['pos_encoder_type']

        # Normalizing the weights
        if ('do_normalization_weights' not in parameters_exp):
            parameters_exp['do_normalization_weights'] = False
        self.do_normalization_weights = parameters_exp['do_normalization_weights']

        # Device
        if ('device' not in parameters_exp):
            parameters_exp['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
        if ('cuda' in parameters_exp['device'].lower()) or ('gpu' in parameters_exp['device']):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # Training params
        self.lr = parameters_exp['lr']
        self.nb_repetitions = parameters_exp['nb_repetitions']
        self.weight_decay = parameters_exp['weight_decay']
        self.batch_size_train = parameters_exp['batch_size_train']
        self.batch_size_test = parameters_exp['batch_size_test']
        self.nb_epochs = parameters_exp['nb_epochs']
        self.criterion = None
        self.current_epoch = 0

        if (parameters_exp['loss_function'].lower() == 'ce'):
            self.criterion = torch.nn.CrossEntropyLoss()
        elif (parameters_exp['loss_function'].lower() == 'gce'):
            self.criterion = GeneralizedCrossEntropy()
        else:
            raise ValueError('Loss function {} is not valid'.format(parameters_exp['loss_function']))


        # Model type
        self.model_to_use = parameters_exp['model_to_use']

        # Some parameters needed to create some models (but not necessarily
        # used by all of the models!)
        if ('d_model' not in parameters_exp):
            parameters_exp['d_model'] = 64

        # Creation of val dataset ?
        if ('separate_val_ds' not in parameters_exp):
            parameters_exp['separate_val_ds'] = True
        self.separate_val_ds = parameters_exp['separate_val_ds']

        # Balanced dataset parameter
        if ('balance_dataset' not in parameters_exp):
            parameters_exp['balance_dataset'] = False
        self.balance_dataset = parameters_exp['balance_dataset']

        # Balance strategy
        if ('balance_strategy' not in parameters_exp):
            parameters_exp['balance_strategy'] = 'undersampling'
        self.balance_strategy = parameters_exp['balance_strategy']

        # Compute class weights parameter
        if ('compute_class_weights' not in parameters_exp):
            parameters_exp['compute_class_weights'] = False
        self.compute_class_weights = parameters_exp['compute_class_weights']

        # Percentage of samples to keep
        if ('percentage_samples_keep' not in parameters_exp):
            parameters_exp['percentage_samples_keep'] = 0.1
        self.percentage_samples_keep = parameters_exp['percentage_samples_keep']

        # Dataset loading
        if (self.dataset_type.lower() == 'esr'):
            # Downloding the dataset if it does not exist
            download_ESR()

            # Parameters for binarization of the dataset
            if ('binarizeDS' not in parameters_exp):
                parameters_exp['binarizeDS'] = True
            # Add channel dim
            if ('add_channel_dim' not in parameters_exp):
                parameters_exp['add_channel_dim'] = False
            else:
                parameters_exp['add_channel_dim'] = string_to_bool(parameters_exp['add_channel_dim'])
            self.add_channel_dim = parameters_exp['add_channel_dim']
            self.parameters_exp = parameters_exp

            # Loading the data
            hdf5_file_path = parameters_exp['dataset_folder'] + '/data.hdf5'
            self.training_data, self.testing_data = loadFromHDF5_EEG(hdf5_file_path)
            if (self.parameters_exp['binarizeDS']):
                self.nb_classes = 2
            else:
                self.nb_classes = 5
            parameters_exp['nb_classes'] = self.nb_classes

            # Splitting the train data into train and validation
            if (self.separate_val_ds):
                train_val_splits = train_val_split_stratified(self.training_data, n_splits=1, test_size=0.2)[0]
                self.training_data, self.val_data = train_val_splits['Train'], train_val_splits['Validation']

            # Keeping only a part of the training data
            if (self.percentage_samples_keep < 1):
                new_training_data = {}
                new_training_data_id = 0
                nb_samples_keep = int(self.percentage_samples_keep*len(self.training_data))
                for i in range(nb_samples_keep):
                    new_training_data[new_training_data_id] = self.training_data[i]
                    new_training_data_id += 1
                self.training_data = new_training_data

            # Balance training dataset (TO BE DONE AFTER NOISE to be realistic)
            # It is done ONLY ON THE TRAINING DATA
            if (self.balance_dataset):
                self.training_data, nb_samples_per_class = balance_dataset(self.training_data, dataset_type=self.dataset_type, balance_strategy=self.balance_strategy)
                print("\nAFTER RESAMPLING we have {} training samples. Number of samples per class: {}".format(len(self.training_data), nb_samples_per_class))

            # Creating the pytorch datasets
            self.train_ds = EEG_EpilepticSeizureRecognition(
                                                data=self.training_data,
                                                feature_type=self.feature_type,
                                                add_channel_dim=self.add_channel_dim,
                                                params=self.parameters_exp
                                             )
            if (self.separate_val_ds):
                self.val_ds = EEG_EpilepticSeizureRecognition(
                                                    data=self.val_data,
                                                    feature_type=self.feature_type,
                                                    add_channel_dim=self.add_channel_dim,
                                                    params=self.parameters_exp
                                                 )
            self.test_ds = EEG_EpilepticSeizureRecognition(
                                                data=self.testing_data,
                                                feature_type=self.feature_type,
                                                add_channel_dim=self.add_channel_dim,
                                                params=self.parameters_exp
                                             )
        elif (self.dataset_type.lower() == 'mnist'):
            # Transformations to apply to the dataset
            transform = torchvision.transforms.Compose(
                                                        [
                                                            torchvision.transforms.CenterCrop(20),
                                                            torchvision.transforms.ToTensor()
                                                        ]
                                                      )

            # Retrieving the training dataset
            self.training_data = torchvision.datasets.MNIST(
                                                                root=parameters_exp['dataset_folder'],
                                                                train=True,
                                                                transform=transform,
                                                                download=True
            )
            # Keeping only a percentage of samples
            print("Original number of training samples (MNIST): {}".format(len(self.training_data)))
            nb_samples_keep = int(self.percentage_samples_keep*len(self.training_data))
            self.training_data = [self.training_data[i] for i in range(len(self.training_data)) if i < nb_samples_keep]
            print('New number of training samples (MNIST): {}'.format(len(self.training_data)))
            # Putting the dataset under the rigt format
            self.training_data = put_MNIST_data_generic_form(self.training_data)

            # Splitting the train data into train and validation
            if (self.separate_val_ds):
                train_val_splits = train_val_split_stratified(self.training_data, n_splits=1, test_size=0.2)[0]
                self.training_data, self.val_data = train_val_splits['Train'], train_val_splits['Validation']

            # Retrieving the test dataset
            self.testing_data = torchvision.datasets.MNIST(
                                                                root=parameters_exp['dataset_folder'],
                                                                train=False,
                                                                transform=transform,
                                                                download=True
                                                        )
            self.testing_data = put_MNIST_data_generic_form(self.testing_data)

            # Balance training dataset (TO BE DONE AFTER NOISE to be realistic)
            # It is done ONLY ON THE TRAINING DATA
            if (self.balance_dataset):
                self.training_data, nb_samples_per_class = balance_dataset(self.training_data, dataset_type=self.dataset_type, balance_strategy=self.balance_strategy)
                print("\nAFTER RESAMPLING we have {} training samples. Number of samples per class: {}".format(len(self.training_data), nb_samples_per_class))

            # Creating the pytorch datasets
            self.train_ds = MNISTDatasetWrapper(data=self.training_data)
            if (self.separate_val_ds):
                self.val_ds = MNISTDatasetWrapper(data=self.val_data)
            self.test_ds = MNISTDatasetWrapper(data=self.testing_data)
        elif (self.dataset_type.lower() == 'fmnist'):
            # Transformations to apply to the dataset
            transform = torchvision.transforms.Compose(
                                                        [
                                                            torchvision.transforms.CenterCrop(20),
                                                            torchvision.transforms.ToTensor()
                                                        ]
                                                      )

            # Retrieving the training dataset
            self.training_data = torchvision.datasets.FashionMNIST(
                                                                    root=parameters_exp['dataset_folder'],
                                                                    train=True,
                                                                    transform=transform,
                                                                    download=True
            )
            # Keeping only a percentage of samples
            print("Original number of training samples (FashionMNIST): {}".format(len(self.training_data)))
            nb_samples_keep = int(self.percentage_samples_keep*len(self.training_data))
            self.training_data = [self.training_data[i] for i in range(len(self.training_data)) if i < nb_samples_keep]
            print('New number of training samples (FashionMNIST): {}'.format(len(self.training_data)))
            # Putting the dataset under the right format
            self.training_data = put_MNIST_data_generic_form(self.training_data)

            # Splitting the train data into train and validation
            if (self.separate_val_ds):
                train_val_splits = train_val_split_stratified(self.training_data, n_splits=1, test_size=0.2)[0]
                self.training_data, self.val_data = train_val_splits['Train'], train_val_splits['Validation']

            # Retrieving the test dataset
            self.testing_data = torchvision.datasets.FashionMNIST(
                                                                    root=parameters_exp['dataset_folder'],
                                                                    train=False,
                                                                    transform=transform,
                                                                    download=True
                                                            )
            self.testing_data = put_MNIST_data_generic_form(self.testing_data)

            # Balance training dataset (TO BE DONE AFTER NOISE to be realistic)
            # It is done ONLY ON THE TRAINING DATA
            if (self.balance_dataset):
                self.training_data, nb_samples_per_class = balance_dataset(self.training_data, dataset_type=self.dataset_type, balance_strategy=self.balance_strategy)
                print("\nAFTER RESAMPLING we have {} training samples. Number of samples per class: {}".format(len(self.training_data), nb_samples_per_class))

            # Creating the pytorch datasets
            self.train_ds = MNISTDatasetWrapper(data=self.training_data)
            if (self.separate_val_ds):
                self.val_ds = MNISTDatasetWrapper(data=self.val_data)
            self.test_ds = MNISTDatasetWrapper(data=self.testing_data)
        elif (self.dataset_type.lower() == 'svhn'):
            # Transformations to apply to the dataset
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(20),
                torchvision.transforms.ToTensor()
            ])

            # Retrieving the training dataset
            self.training_data = torchvision.datasets.SVHN(
                root=parameters_exp['dataset_folder'],
                split='train',
                transform=transform,
                download=True
            )

            # Keeping only a percentage of samples
            print("Original number of training samples (SVHN): {}".format(len(self.training_data)))
            nb_samples_keep = int(self.percentage_samples_keep * len(self.training_data))
            self.training_data = [self.training_data[i] for i in range(len(self.training_data)) if i < nb_samples_keep]
            print('New number of training samples (SVHN): {}'.format(len(self.training_data)))

            # Putting the dataset under the right format
            self.training_data = put_SVHN_data_generic_form(self.training_data)

            # Splitting the train data into train and validation
            if (self.separate_val_ds):
                train_val_splits = train_val_split_stratified(self.training_data, n_splits=1, test_size=0.2)[0]
                self.training_data, self.val_data = train_val_splits['Train'], train_val_splits['Validation']

            # Retrieving the test dataset
            self.testing_data = torchvision.datasets.SVHN(
                root=parameters_exp['dataset_folder'],
                split='test',
                transform=transform,
                download=True
            )
            self.testing_data = put_SVHN_data_generic_form(self.testing_data)

            # Balance training dataset (TO BE DONE AFTER NOISE to be realistic)
            # It is done ONLY ON THE TRAINING DATA
            if (self.balance_dataset):
                self.training_data, nb_samples_per_class = balance_dataset(self.training_data, dataset_type=self.dataset_type, balance_strategy=self.balance_strategy)
                print("\nAFTER RESAMPLING we have {} training samples. Number of samples per class: {}".format(len(self.training_data), nb_samples_per_class))

            # Creating the pytorch datasets
            self.train_ds = SVHNDatasetWrapper(data=self.training_data)
            if (self.separate_val_ds):
                self.val_ds = MNISTDatasetWrapper(data=self.val_data)
            self.test_ds = SVHNDatasetWrapper(data=self.testing_data)
            
        elif (self.dataset_type.lower() == 'emnist'):
            # Transformations to apply to the dataset
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(20),
                torchvision.transforms.ToTensor()
            ])

            # Retrieving the training dataset
            self.training_data = torchvision.datasets.EMNIST(
                root=parameters_exp['dataset_folder'],
                split='letters',
                train=True,
                transform=transform,
                download=True
            )

            # Keeping only a percentage of samples
            print("Original number of training samples (EMNIST): {}".format(len(self.training_data)))
            nb_samples_keep = int(self.percentage_samples_keep * len(self.training_data))
            self.training_data = [self.training_data[i] for i in range(len(self.training_data)) if i < nb_samples_keep]
            print('New number of training samples (EMNIST): {}'.format(len(self.training_data)))

            # Putting the dataset under the right format
            self.training_data = put_EMNIST_data_generic_form(self.training_data)

            # Splitting the train data into train and validation
            if (self.separate_val_ds):
                train_val_splits = train_val_split_stratified(self.training_data, n_splits=1, test_size=0.2)[0]
                self.training_data, self.val_data = train_val_splits['Train'], train_val_splits['Validation']

            # Retrieving the test dataset
            self.testing_data = torchvision.datasets.EMNIST(
                root=parameters_exp['dataset_folder'],
                split='letters',
                train=False,
                transform=transform,
                download=True
            )
            self.testing_data = put_EMNIST_data_generic_form(self.testing_data)

            # Balance training dataset (TO BE DONE AFTER NOISE to be realistic)
            # It is done ONLY ON THE TRAINING DATA
            if (self.balance_dataset):
                self.training_data, nb_samples_per_class = balance_dataset(self.training_data, dataset_type=self.dataset_type, balance_strategy=self.balance_strategy)
                print("\nAFTER RESAMPLING we have {} training samples. Number of samples per class: {}".format(len(self.training_data), nb_samples_per_class))

            # Creating the pytorch datasets
            self.train_ds = EMNISTDatasetWrapper(data=self.training_data)
            if (self.separate_val_ds):
                self.val_ds = EMNISTDatasetWrapper(data=self.val_data)
            self.test_ds = EMNISTDatasetWrapper(data=self.testing_data)
        elif (self.dataset_type.lower() == 'kmnist'):

            # Transformations to apply to the dataset
            transform = torchvision.transforms.Compose(
                                                        [
                                                            torchvision.transforms.CenterCrop(20),
                                                            torchvision.transforms.ToTensor()
                                                        ]
                                                      )

            # Retrieving the training dataset
            self.training_data = torchvision.datasets.KMNIST(
                                                                root=parameters_exp['dataset_folder'],
                                                                train=True,
                                                                transform=transform,
                                                                download=True
            )
            # Keeping only a percentage of samples
            print("Original number of training samples (KMNIST): {}".format(len(self.training_data)))
            nb_samples_keep = int(self.percentage_samples_keep*len(self.training_data))
            self.training_data = [self.training_data[i] for i in range(len(self.training_data)) if i < nb_samples_keep]
            print('New number of training samples (KMNIST): {}'.format(len(self.training_data)))
            # Putting the dataset under the right format
            self.training_data = put_MNIST_data_generic_form(self.training_data)

            # Splitting the train data into train and validation
            if (self.separate_val_ds):
                train_val_splits = train_val_split_stratified(self.training_data, n_splits=1, test_size=0.2)[0]
                self.training_data, self.val_data = train_val_splits['Train'], train_val_splits['Validation']

            # Retrieving the test dataset
            self.testing_data = torchvision.datasets.KMNIST(
                                                                root=parameters_exp['dataset_folder'],
                                                                train=False,
                                                                transform=transform,
                                                                download=True
                                                        )
            self.testing_data = put_MNIST_data_generic_form(self.testing_data)

            # Balance training dataset (TO BE DONE AFTER NOISE to be realistic)
            # It is done ONLY ON THE TRAINING DATA
            if (self.balance_dataset):
                self.training_data, nb_samples_per_class = balance_dataset(self.training_data, dataset_type=self.dataset_type, balance_strategy=self.balance_strategy)
                print("\nAFTER RESAMPLING we have {} training samples. Number of samples per class: {}".format(len(self.training_data), nb_samples_per_class))

            # Creating the pytorch datasets
            self.train_ds = MNISTDatasetWrapper(data=self.training_data)
            if (self.separate_val_ds):
                self.val_ds = MNISTDatasetWrapper(data=self.val_data)
            self.test_ds = MNISTDatasetWrapper(data=self.testing_data)
        else:
            raise ValueError('Dataset type {} is not supported'.format(self.dataset_type))
        print("Number of samples in the training dataset: ", len(self.train_ds))
        if (self.separate_val_ds):
            print("Number of samples in the validation dataset: ", len(self.val_ds))
        print("Number of samples in the testing dataset: ", len(self.test_ds))

        # Determining the audio shape for the selected time-frequency representation
        sample, label = self.train_ds[0]
        self.audio_feature_shape = sample.shape
        print("Shape of the used representation: {}".format(self.audio_feature_shape))

        # Parameters of the exp
        self.parameters_exp = parameters_exp

    def compute_dataset_mean_std(self):
        """
            Computes the mean and standard deviation of the WHOLE dataset
            for normalization purposes
        """
        # Creating dataloaders
        self.dataloadersCreation()

        # Getting all the samples
        samples = []
        for batch, label in self.train_loader:
            for sample in batch:
                samples.append(sample.cpu().detach().numpy())
        if (self.separate_val_ds):
            for batch, label in self.val_loader:
                for sample in batch:
                    samples.append(sample.cpu().detach().numpy())
        samples = np.array(samples)
        self.dataset_mean = np.mean(np.array(samples))
        self.dataset_std = np.std(samples)

    def normalize_dataset(self):
        """
            Normalize the dataset by substracting the mean and dividing by
            the std
        """
        # Creating the pytorch datasets
        if (self.dataset_type.lower() == 'esr'):
            # NO NEED TO NORMALIZE IT, IT WAS DONE WHEN CREATING THE HDF5 FILE !!!!
            self.train_ds = EEG_EpilepticSeizureRecognition(
                                                data=self.training_data,
                                                feature_type=self.feature_type,
                                                add_channel_dim=self.add_channel_dim,
                                                params=self.parameters_exp
                                             )
            if (self.separate_val_ds):
                self.val_ds = EEG_EpilepticSeizureRecognition(
                                                    data=self.val_data,
                                                    feature_type=self.feature_type,
                                                    add_channel_dim=self.add_channel_dim,
                                                    params=self.parameters_exp
                                                 )
                self.test_ds = EEG_EpilepticSeizureRecognition(
                                                    data=self.testing_data,
                                                    feature_type=self.feature_type,
                                                    add_channel_dim=self.add_channel_dim,
                                                    params=self.parameters_exp
                                                 )
        elif (self.dataset_type.lower() == 'mnist'):
            pass
        elif (self.dataset_type.lower() == 'fmnist'):
            pass
        elif (self.dataset_type.lower() == 'kmnist'):
            pass
        elif (self.dataset_type.lower() == 'svhn'):
            pass        
        else:
            raise ValueError('Dataset type {} is not supported'.format(self.dataset_type))
        print("Number of samples in the training dataset: ", len(self.train_ds))
        if (self.separate_val_ds):
            print("Number of samples in the val dataset: ", len(self.val_ds))
        print("Number of samples in the testing dataset: ", len(self.test_ds))

    def dataloadersCreation(self):
        """
            Create the train and test dataloader necessary to train and test a
            CNN classification model
        """
        # Training set
        train_indices = list(range(0, len(self.train_ds)))
        train_sampler = SubsetRandomSampler(train_indices)
        # train_sampler = SequentialSampler(train_indices)
        self.train_loader = torch.utils.data.DataLoader(self.train_ds,\
                                                       batch_size=self.batch_size_train,\
                                                       sampler=train_sampler)

        # Validation set
        if (self.separate_val_ds):
            val_indices = list(range(0, len(self.val_ds)))
            val_sampler = SubsetRandomSampler(val_indices)
            # val_sampler = SequentialSampler(val_indices)
            self.val_loader = torch.utils.data.DataLoader(self.val_ds,\
                                                           batch_size=self.batch_size_train,\
                                                           sampler=val_sampler)

        # Testing set
        test_indices = list(range(0, len(self.test_ds)))
        test_sampler = SubsetRandomSampler(test_indices)
        # test_sampler = SequentialSampler(test_indices)
        self.test_loader = torch.utils.data.DataLoader(self.test_ds,\
                                                       batch_size=self.batch_size_test,\
                                                       sampler=test_sampler)

    def modelCreation(self):
        """
            Creates a model to be trained on the selected time-frequency
            representation
        """
        # Creating the model
        if (self.model_type.lower() == '2dcnn'):
            if (self.model_to_use.lower() in ['mnist2dcnn','fmnist2dcnn']):
                self.model = MnistClassificationModel(input_channels=1, nb_classes=10)
                pass
            elif (self.model_to_use.lower() in ['kmnistresnet18','fmnistresnet18']):
                self.model = ResNet18ClassificationModel(input_channels=1,nb_classes=10)
            elif (self.model_to_use.lower() in ['svhnresnet18']):
                self.model = ResNet18ClassificationModel(input_channels=3,nb_classes=10)

            elif (self.model_to_use.lower() == 'timefrequency2dcnn'):
                self.nb_init_filters = self.parameters_exp['nb_init_filters']
                self.increase_nb_filters_mode = self.parameters_exp['increase_nb_filters_mode']
                self.pooling_mode = self.parameters_exp['pooling_mode']
                self.dropout_probability = self.parameters_exp['dropout_probability']
                self.model = TimeFrequency2DCNN(nb_init_filters=self.nb_init_filters,
                                 increase_nb_filters_mode=self.increase_nb_filters_mode,
                                 pooling_mode=self.pooling_mode,
                                 dropout_probability=self.dropout_probability,
                                 input_shape=self.audio_feature_shape,
                                 num_classes=self.nb_classes)
            else:
                raise ValueError("Model to use {} is not valid".format(self.model_to_use))


        elif (self.model_type.lower() == 'transformer'):
            if (self.model_to_use.lower() == 'rawaudiomultichannelcnn'):
                print("=======> USING RAWAUDIOMULTICHANNELCNN TRANSFORMER\n")
                self.d_model = self.parameters_exp['d_model']
                self.model = model = TransformerClassifierMultichannelCNN(
                                                                self.parameters_exp['in_channels'],
                                                                self.parameters_exp['nhead'],
                                                                self.parameters_exp['d_hid'],
                                                                self.parameters_exp['nlayers'],
                                                                self.parameters_exp['dropout'],
                                                                self.parameters_exp['nb_features_projection'],
                                                                self.parameters_exp['d_model'],
                                                                self.nb_classes,
                                                                self.parameters_exp['classification_pool'],
                                                                self.parameters_exp['n_conv_layers'],
                                                                self.parameters_exp['pos_encoder_type']
                                                            )
            else:
                raise ValueError("Transformer type {} is not valid".format(self.model_to_use))
        else:
            raise ValueError("Model type {} is not valid".format(self.model_type))


        # Sending the model to the correct device
        self.model.to(self.device)

    def balance_classes_loss(self):
        # Getting the labels for the training set
        y_train = np.array([self.train_ds[sample_id][1] for sample_id in range(len(self.train_ds))])

        # Computing the weights
        if (self.compute_class_weights):
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        else:
            class_weights = np.array([1.0 for _ in range(len(np.unique(y_train)))])
        print("\n\nClass weights: {}\n\n".format(class_weights))
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        class_weights = class_weights.to(self.device)

        # Creatining the new weighthed loss
        if (self.parameters_exp['loss_function'].lower() == 'ce'):
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        elif (self.parameters_exp['loss_function'].lower() == 'gce'):
            self.criterion = GeneralizedCrossEntropy(class_weights=class_weights)
            # raise NotImplementedError("Class weighting it is not implemented for GCE loss function")
        else:
            raise ValueError('Loss function {} is not valid'.format(self.parameters_exp['loss_function']))

    def createOptimizer(self, model_params_dict):
        """
            Creation of the optimizer(s)
        """
        # Model parameters
        model_parameters = model_params_dict['All']
        # Creating the optimizer
        if (self.model_type.lower() != 'transformer'):
            # Creating the optimizer
            if (self.model_to_use.lower() == 'mnist2dcnn'):
                self.optimizer = torch.optim.Adamax(model_parameters, lr=self.lr, weight_decay=self.weight_decay)
            else:
                self.optimizer = torch.optim.Adam(model_parameters, lr=self.lr, weight_decay=self.weight_decay)

            # Creating the learning rate scheduler for the global optimizer
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',\
                                                               factor=0.1, patience=5,\
                                                               threshold=1e-4, threshold_mode='rel',\
                                                               cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        else:
            self.optimizer = noam.Noam(
                            params=model_parameters,
                            lr=self.lr,
                            betas=(0.9, 0.999),
                            eps=1e-16,
                            optimized_update=True,
                            amsgrad=False,
                            warmup=self.warmup,
                            d_model=self.d_model,
                        )


    def compute_forward_pass(self, batch, epoch_nb, keep_grad=True):
        # Getting the data and the labels
        audio_features, labels = batch
        audio_features, labels = audio_features.to(self.device), labels.to(self.device)

        # Computing the loss
        out = self.model(audio_features) # Generate predictions
        loss = self.criterion(out, labels.long()) # Calculate loss

        # Getting the HARD predictions
        y_true, y_pred = [], []
        for i in range(len(out)):
            true_class = int(labels[i])
            y_true.append(true_class)
            predicted_class = int(out[i].max(0)[1])
            y_pred.append(predicted_class)

        predictions = {
                        'TrueLabels': y_true,
                        'PredictedLabels': y_pred,
                    }

        return loss, predictions

    def optimize_step(self, loss_value):
        """
            Simple optimization step
        """
        # Zero grad for all optimizers
        self.optimizer.zero_grad()

        # Gradients for the quantized model
        loss_value.backward()

        # Update the parameters
        self.optimizer.step()

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
        with torch.no_grad():
            for named_param in self.model.named_parameters():
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
        # Creating the dataloaders
        self.dataloadersCreation()

        # Creating the model
        self.modelCreation()

        # Creating the optimizer
        model_params_dict = {'All': self.model.parameters()}
        self.createOptimizer(model_params_dict)

    def optimize_hyperparams(self):
        pass

    def apply_lr_sched(self, mean_loss_val_fixed_epoch=None):
        """
            Applies the learning rate scheduler(s)
        """
        if (self.model_type.lower() != 'transformer') and (mean_loss_val_fixed_epoch is not None):
            self.sched.step(mean_loss_val_fixed_epoch) # For ReduceLROnPlateau

    def single_train(self):
        """
            Trains a model one time during self.nb_epochs epochs
        """
        # Initialization
        self.init_single_train()

        # Data structures for the losses and the predictions
        loss_values = {
                        'Train': [0 for _ in range(self.nb_epochs)],
                        'Val': [0 for _ in range(self.nb_epochs)],
                        'Test': [0 for _ in range(self.nb_epochs)]
                      }
        predictions_results = {}
        for dataset_split in ['Train', 'Val', 'Test']:
            predictions_results[dataset_split] = {}
            for type_labels in ['TrueLabels', 'PredictedLabels']:
                predictions_results[dataset_split][type_labels] =  [[] for _ in range(self.nb_epochs)]

        # Epochs
        sparsity_rates_per_epoch = []
        test_mcc_per_epoch = []
        for epoch in tqdm(range(self.nb_epochs)):
            # Hyperparams optimization usin Simulated Annealing
            if (self.do_hyperparams_opt) and (epoch == 0):
                self.optimize_hyperparams()

            # Training
            self.model.train()
            tmp_train_losses = []
            for batch in self.train_loader:
                # Normalizing the weights
                if (self.do_normalization_weights):
                    # self.normalize_weights(per_channel_norm=True)
                    self.normalize_weights(per_channel_norm=False)

                # Forward pass
                train_loss, train_predictions = self.compute_forward_pass(batch, epoch, keep_grad=True)
                tmp_train_losses.append(train_loss.detach().data.cpu().numpy())

                # Updating the weights
                self.optimize_step(loss_value=train_loss)

                # Updating the predictions results of the current epoch
                predictions_results['Train']['TrueLabels'][epoch] += train_predictions['TrueLabels']
                predictions_results['Train']['PredictedLabels'][epoch] += train_predictions['PredictedLabels']
            loss_values['Train'][epoch] = np.mean(tmp_train_losses)

            # Validation
            if (self.separate_val_ds):
                with (torch.no_grad()):
                    self.model.eval()
                    tmp_val_losses = []
                    for batch in self.val_loader:
                        # Forward pass
                        val_loss, val_predictions = self.compute_forward_pass(batch, epoch, keep_grad=False)
                        tmp_val_losses.append(val_loss.detach().data.cpu())

                        # Updating the predictions results of the current epoch
                        predictions_results['Val']['TrueLabels'][epoch] += val_predictions['TrueLabels']
                        predictions_results['Val']['PredictedLabels'][epoch] += val_predictions['PredictedLabels']

                    loss_values['Val'][epoch] = np.mean(tmp_val_losses)

                    # Applying learning rate scheduler
                    self.apply_lr_sched(loss_values['Val'][epoch])
            else:
                # Applying learning rate scheduler
                self.apply_lr_sched()

            # Testing
            with (torch.no_grad()):
                self.model.eval()
                tmp_test_losses = []
                for batch in self.test_loader:
                    # Forward pass
                    test_loss, test_predictions = self.compute_forward_pass(batch, epoch, keep_grad=False)
                    tmp_test_losses.append(test_loss.detach().data.cpu())

                    # Updating the predictions results of the current epoch
                    predictions_results['Test']['TrueLabels'][epoch] += test_predictions['TrueLabels']
                    predictions_results['Test']['PredictedLabels'][epoch] += test_predictions['PredictedLabels']

                loss_values['Test'][epoch] = np.mean(tmp_test_losses)
                print("================================================================================")
                print("METRICS\n")
                print("\n=======>Train loss at epoch {} is {}".format(epoch, loss_values['Train'][epoch]))
                if (self.separate_val_ds):
                    print("\t\tVal loss at epoch {} is {}".format(epoch, loss_values['Val'][epoch]))
                print("\t\tTest loss at epoch {} is {}".format(epoch, loss_values['Test'][epoch]))
                print("\n=======>Train F1 Score at epoch {} is {}\n".format(epoch, f1_score(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch], average='macro')))
                if (self.separate_val_ds):
                    print("\t\tVal F1 Score at epoch {} is {}".format(epoch, f1_score(predictions_results['Val']['TrueLabels'][epoch], predictions_results['Val']['PredictedLabels'][epoch], average='micro')))
                print("\t\tTest F1 Score at epoch {} is {}".format(epoch, f1_score(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch], average='micro')))
                print("\n=======>Train accuracy at epoch {} is {}\n".format(epoch, accuracy_score(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch])))
                if (self.separate_val_ds):
                    print("\t\tVal accuracy at epoch {} is {}".format(epoch, accuracy_score(predictions_results['Val']['TrueLabels'][epoch], predictions_results['Val']['PredictedLabels'][epoch])))
                print("\t\tTest accuracy at epoch {} is {}".format(epoch, accuracy_score(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch])))
                print("\n=======>Train MCC at epoch {} is {}\n".format(epoch, matthews_corrcoef(predictions_results['Train']['TrueLabels'][epoch], predictions_results['Train']['PredictedLabels'][epoch])))
                if (self.separate_val_ds):
                    print("\t\tVal MCC at epoch {} is {}".format(epoch, matthews_corrcoef(predictions_results['Val']['TrueLabels'][epoch], predictions_results['Val']['PredictedLabels'][epoch])))
                print("\t\tTest MCC at epoch {} is {}".format(epoch, matthews_corrcoef(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch])))
                print("================================================================================\n\n")
                test_mcc_per_epoch.append(  matthews_corrcoef(predictions_results['Test']['TrueLabels'][epoch], predictions_results['Test']['PredictedLabels'][epoch]) )
            # Saving the model and the current results
            current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")

            # Sparsity rate at the given epoch
            self.non_zero_params = self.countNonZeroWeights(self.model)
            nb_total_params, nb_params_to_quantize = self.get_nb_params_to_quantize()
            if (not self.countNonZeroParamsQuantizedLayers):
                sparsity_rate = (nb_total_params-self.non_zero_params)/nb_params_to_quantize
            else:
                print('!!!!Counting the non zero parameters of ONLY THE LAYERS TO QUANTIZE ({} params to quantize)'.format(nb_params_to_quantize))
                sparsity_rate = 1-(self.non_zero_params/nb_params_to_quantize)
            sparsity_rates_per_epoch.append(sparsity_rate.detach().cpu().numpy())
            print("=======> SPARSITY RATE AT EPOCH {}: {}\n\n\n".format(epoch, sparsity_rate*100))
            self.current_epoch += 1

        return {'Loss': loss_values, 'Predictions': predictions_results, 'SparsityRatePerEpoch': sparsity_rates_per_epoch,'TestMccPerEpoch':test_mcc_per_epoch}

    def countNonZeroWeights(self, model):
        """
            Count the number of non zero parameters in the model

            Arguments:
            ----------
            model: torch model
                Model from which we want to count the weights.
        """
        nonzero = 0
        for name, param in model.named_parameters():
            nonzero += torch.count_nonzero(param)

        return nonzero

    def get_nb_params_to_quantize(self):
        nb_params_to_quantize = 0
        nb_total_params = 0
        for n, p in self.model.named_parameters():
            # Nb params layer
            nb_params_layer = 1
            for val in p.shape:
                nb_params_layer *= val

            # Nb tot params
            nb_total_params += nb_params_layer

        nb_params_to_quantize = 0

        return nb_total_params, nb_params_to_quantize

    def holdout_train(self):
        """
            Does a holdout training repeated self.nb_repetitions times
        """
        repetitionsResults = {}
        for nb_repetition in range(self.nb_repetitions):
            print("\n\n=======> Repetitions {} <=======".format(nb_repetition))
            # Doing single train
            tmp_results = self.single_train()
            repetitionsResults[nb_repetition] = tmp_results

            # Saving the final model and the results
            # Model
            torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'model': self.model
                        }, self.results_folder + '/model/final_model-{}_rep-{}.pth'.format(self.exp_id, nb_repetition))
            # Results
            with open(self.results_folder + '/metrics/results_exp-{}_rep-{}.pth'.format(self.exp_id, nb_repetition), "wb") as fp:   #Pickling
                pickle.dump(tmp_results, fp)


        # Saving the results of the different repetitions
        with open(self.results_folder + '/metrics/final_results_all_repetitions.pth', "wb") as fp:   #Pickling
            pickle.dump(repetitionsResults, fp)

        # Need to save them 

        #save_file_name = "./saves/results_d{dataset}_m{model}_e{exp_id}"
        
        ''' DEBUG logs
        for rep_key in repetitionsResults.keys():
            for e_key in repetitionsResults[rep_key].keys():

                print(f"==== [{rep_key}] key : ", e_key )
                print(f"==== [{rep_key}] val : ",repetitionsResults[rep_key][e_key]) '''

    def gridSearch(self):
        """
            Does a grid search for some hyper-parameters
        """
        # Defining the values of the parameters to test
        lr_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

        # Iterating over the different values of the hyper-parameters
        base_results_folder = self.results_folder
        for lr in lr_values:
            # Updating the hyper-paramet of the experiment
            self.lr = lr

            # Creating the datasets folder
            current_datetime = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
            os.mkdir(base_results_folder + '/LR-{}_{}/'.format(self.lr, current_datetime))
            os.mkdir(base_results_folder + '/LR-{}_{}/model/'.format(self.lr, current_datetime))
            os.mkdir(base_results_folder + '/LR-{}_{}/metrics/'.format(self.lr, current_datetime))
            self.results_folder = base_results_folder + '/LR-{}_{}/'.format(self.lr, current_datetime)

            # Training
            self.holdout_train()

        self.results_folder = base_results_folder

    def setResultsFolder(self, results_folder):
        """
            Set the folder where the results are going to be stored
        """
        self.results_folder = results_folder

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
    # torch.use_deterministic_algorithms(True) # For reproducibility purposes

    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    default_parameters_file = "../../parameters_files/MNIST/mnist_FP.json"
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
    resultsFolder = './results/' + parameters_exp['exp_id'] + '_' + 'OW'
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
            shutil.copy2('./src/Models/CNNs/time_frequency_simple_CNN.py', resultsFolder + '/params_exp/network_architecture.py')
        elif (parameters_exp['model_to_use'].lower() == 'mnist2dcnn'):
            shutil.copy2('./src/Models/CNNs/mnist_CNN.py', resultsFolder + '/params_exp/network_architecture.py')
        elif (parameters_exp['model_to_use'].lower() == 'fmnist2dcnn'):
            shutil.copy2('./src/Models/CNNs/mnist_CNN.py', resultsFolder + '/params_exp/network_architecture.py')   
        elif (parameters_exp['model_to_use'].lower() in ['kmnistresnet18','fmnistresnet18','svhnresnet18']):
            shutil.copy2('./src/Models/CNNs/resnet18.py', resultsFolder + '/params_exp/network_architecture.py')    
        else:
            raise ValueError('2D CNN {} is not valid'.format(parameters_exp['model_to_use']))

    elif (parameters_exp['model_type'].lower() == 'transformer'):
        if (parameters_exp['model_to_use'].lower() == 'rawaudiomultichannelcnn'):
            shutil.copy2('./src/Models/Transformers/Transformer_Encoder_RawAudioMultiChannelCNN.py', resultsFolder + '/params_exp/network_architecture.py')
        else:
            raise ValueError("Transformer type {} is not valid".format(parameters_exp['model_to_use']))
    else:
        raise ValueError("Model type {} is not valid".format(parameters_exp['model_type']))

    # Save the data distribution
    if (exp.dataset_type.lower() == 'hits'):
        shutil.copy2(parameters_exp['dataset_folder'] + '/data.hdf5', resultsFolder + '/params_exp/data.hdf5')


    #==========================================================================#
    print("\n\n==================== End of the experiment ====================\n\n")



if __name__=="__main__":
    main()
