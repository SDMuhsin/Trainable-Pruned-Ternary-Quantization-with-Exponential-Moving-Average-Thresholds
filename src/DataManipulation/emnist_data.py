#!/usr/bin/env python3
"""
    Code for the EMNIST dataset
"""
import torch
import torchvision
from torch.utils.data import Dataset

def put_EMNIST_data_generic_form(torchvision_emnist_data):
    """
        Put the data of a torchvision EMNIST dataset under the same format as
        the rest of the datasets (HITS, ECG, ESR)

        Arguments:
        ----------
        torchvision_emnist_data: torchvision.datasets.emnist.EMNIST

        Returns:
        --------
        generic_EMNIST_data: dict
            Dict where the keys are the ids of the samples and the values are
            also dictionaries with two keys: 'Data' and 'Label'
    """
    generic_EMNIST_data = {}
    for id_current_sample in range(len(torchvision_emnist_data)):
        generic_EMNIST_data[id_current_sample] = {
                                                    'Data': torchvision_emnist_data[id_current_sample][0],
                                                    'Label': torchvision_emnist_data[id_current_sample][1]
                                                }

    return generic_EMNIST_data

class EMNISTDatasetWrapper(Dataset):
    """
        EMNIST dataset wrapper (for the EMNIST torchvision dataset)

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
