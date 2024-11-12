#!/usr/bin/env python3
"""
    Code for the MNIST dataset
"""
import torch
import torchvision
from torch.utils.data import Dataset


def put_SVHN_data_generic_form(torchvision_svhn_data):
    """
    Put the data of a torchvision SVHN dataset under the same format as
    the rest of the datasets (HITS, ECG, ESR)
    Arguments:
    ----------
    torchvision_svhn_data: torchvision.datasets.SVHN
    Returns:
    --------
    generic_SVHN_data: dict
        Dict where the keys are the ids of the samples and the values are
        also dictionaries with two keys: 'Data' and 'Label'
    """
    generic_SVHN_data = {}
    for id_current_sample in range(len(torchvision_svhn_data)):
        generic_SVHN_data[id_current_sample] = {
            'Data': torchvision_svhn_data[id_current_sample][0],
            'Label': torchvision_svhn_data[id_current_sample][1]
        }
    return generic_SVHN_data

class SVHNDatasetWrapper(Dataset):
    """
    SVHN dataset wrapper (for the SVHN torchvision dataset)
    Argument:
    ---------
    data: dict
        Dict where the keys are the ids of the samples and the values are
        also dictionaries with two keys: 'Data' and 'Label'
    """
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # Getting the sample
        sample_data, label = self.data[i]['Data'], self.data[i]['Label']
        return sample_data, label
