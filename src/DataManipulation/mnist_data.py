#!/usr/bin/env python3
"""
    Code for the MNIST dataset
"""
import torch
import torchvision
from torch.utils.data import Dataset


def put_MNIST_data_generic_form(torchvision_mnist_data):
    """
        Put the data of a torchvision MNIST dataset under the same format as
        the rest of the datasets (HITS, ECG, ESR)

        Arguments:
        ----------
        torchvision_mnist_data: torchvision.datasets.mnist.MNIST

        Returns:
        --------
        generic_MNIST_data: dict
            Dict where the keys are the ids of the samples and the values are
            also dictionaries with two keys: 'Data' and 'Label'
    """
    generic_MNIST_data = {}
    for id_current_sample in range(len(torchvision_mnist_data)):
        generic_MNIST_data[id_current_sample] = {
                                                    'Data': torchvision_mnist_data[id_current_sample][0],
                                                    'Label': torchvision_mnist_data[id_current_sample][1]
                                                }

    return generic_MNIST_data

class MNISTDatasetWrapper(Dataset):
    """
        MNIST dataset wrapper (for the MNIST torchvision dataset)

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
