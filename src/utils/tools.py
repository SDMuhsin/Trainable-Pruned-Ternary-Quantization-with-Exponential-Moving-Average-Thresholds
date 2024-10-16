#!/usr/bin/env python3

import os
import csv
import h5py

from collections import Counter

from random import shuffle

import numpy as np

import torch

from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def change_data_samples_path(data_dict, new_root_path, dataset_type='HITS'):
    """
        Change the paths of the "ImagePath" and "RawDataPath" values of the
        samples in data_dict

        Arguments:
        ----------
        data_dict: dict
            Dictionary where the keys are the samples IDs and the values are the
            samples. Each sample is also represented by a dictionary having
            different keys (including "ImagePath" and "RawDataPath")
        dataset_type: str
            Type of dataset to change the root path of the samples. Two
            options: HITS or SyntheticSoftLabels
    """
    new_data_dict = {}
    for sampleID in data_dict:
        sample = data_dict[sampleID]
        image_path = sample['ImagePath'].split('/')
        if (dataset_type.lower() == 'hits'):
            raw_data_path = sample['RawDataPath'].split('/')
        root_path_idx = 0
        while (image_path[root_path_idx] != 'data'):
            root_path_idx += 1
        new_sample = {}
        for key in sample:
            if (key == 'ImagePath'):
                value = new_root_path + '/'.join(image_path[root_path_idx:])
            elif (key == 'RawDataPath'):
                value = new_root_path + '/'.join(raw_data_path[root_path_idx:])
            else:
                value = sample[key]
            new_sample[key] = value
        new_data_dict[sampleID] = new_sample

    return new_data_dict


def string_to_bool(string):
    if (type(string) == bool):
        return string
    else:
        if (string.lower() == 'true'):
            return True
        elif (string.lower() == 'false'):
            return False
        else:
            raise ValueError("String {} is not valid to be transformed into boolean".format(string))

def train_val_split_stratified(data, test_size, n_splits, is_hits_dataset=False):
    """
        Split the data into two stratified splits, one for training and one for
        validation.

        Arguments:
        ----------
        data: dict
            Containing the different samples of the dataset.
        test_size: float
            Percentage of samples to be used as test.
        n_splits: int
            Number of splits to create. Interesting if cross-validation is used
            as evaluation metric
        is_hits_dataset: bool
            True if the data come from a HITS dataset. This is necessary as
            the labels of HITS datasets are probabilities of belonging to
            a class and not hard labels

        Returns:
        --------
        splits: list
            List where each element correspond to a train/validation split.
            Each element is a dict with two keys, 'Train' and 'Validation'
            corresponding to the train and validation samples
    """
    # Getting the indices of the samples and the labels. This will allow us
    # to easily create the train and validation splits
    samples_ids = []
    samples_labels = []
    for sample_id in data:
        # Adding the sample ID
        samples_ids.append(sample_id)

        # Getting the label
        if (is_hits_dataset):
            softLabelSample = {"A": data[sample_id]['ArtefactScore'], "EG": data[sample_id]['EgScore'], "ES": data[sample_id]['EsScore']}
            label = getHardLabelSingleSample(softLabelSample)
        else:
            label = data[sample_id]['Label']
        samples_labels.append(label)

    # Creating the instance of stratified sampling
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

    # Getting the different splits
    splits = []
    for train_idxs, val_idxs in sss.split(samples_ids, samples_labels):
        # Creating the variable for the current split
        split = {'Train': {}, 'Validation': {}}

        # Getting the training samples
        new_train_id = 0
        for train_idx in train_idxs:
            train_sample_id = samples_ids[train_idx]
            split['Train'][new_train_id] = data[train_sample_id]
            new_train_id += 1

        # Getting the validation samples
        new_val_id = 0
        for val_idx in val_idxs:
            val_sample_id = samples_ids[val_idx]
            split['Validation'][new_val_id] = data[val_sample_id]
            new_val_id += 1

        # Adding the split to the list of splits
        splits.append(split)

    return splits


def balance_dataset(data, dataset_type='HITS', balance_strategy='Undersampling'):
    """
        Balance a dataset by undersampling/oversampling the majority class

        Arguments:
        ----------
        data: dict
            Dictionary where the keys are the samples' IDs and the values are
            also dictionaries. All the datasets besides the HITS ones have
            at least two keys: 'Data' and 'Label'.
            The HITS datasets samples' have different keys (see function
            loadFromHDF5 from src.DataManipulation.my_data)
        dataset_type: str
            Type of the dataset to balance
        balance_strategy: str
            Strategy used to balance the dataset
    """
    # Fixing the random state generator
    random_seed = 42

    # Creating an X and y arrays to used sklearn under and oversampling functions
    X, y = [], []
    for sample_id in data:
        # Storing the sample ID
        X.append([sample_id])

        # Getting the label
        if (dataset_type.lower() == 'hits'):
            softLabelSample = {"A": data[sample_id]['ArtefactScore'], "EG": data[sample_id]['EgScore'], "ES": data[sample_id]['EsScore']}
            label = getHardLabelSingleSample(softLabelSample)
        elif (dataset_type.lower() == 'syntheticsoftlabels'):
            label = data[sample_id]['HardLabel']
        else:
            label = data[sample_id]['Label']

        # Storing the label
        y.append(label)

    # Re-sampling the data
    if (balance_strategy.lower() == 'undersampling'):
        rus = RandomUnderSampler(random_state=0, sampling_strategy='not minority')
        X_resampled, y_resampled = rus.fit_resample(X, y)
    elif (balance_strategy.lower() == 'oversampling'):
        ros = RandomUnderSampler(random_state=0, sampling_strategy='not majority')
        X_resampled, y_resampled = ros.fit_resample(X, y)
    else:
        raise NotImplementedError("Balance dataset strategy {} is not implemented".format(balance_strategy))

    # Number of samples per class
    nb_samples_per_class = Counter(y_resampled)

    # Creating a new list of samples
    balanced_data_list = []
    for element in X_resampled:
        # Sample id
        sample_id = element[0]

        # Adding the sample to the new balanced data
        balanced_data_list.append(data[sample_id])
    shuffle(balanced_data_list)

    # Creating the final balanced data dictionary
    balanced_data = {i:balanced_data_list[i] for i in range(len(balanced_data_list))}

    return balanced_data, nb_samples_per_class
