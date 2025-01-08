#!/usr/bin/env python3
"""
    Code for the Pascal VOC dataset
"""
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

def put_PascalVOC_data_generic_form(torchvision_pascal_data):
    """
        Put the data of a torchvision Pascal VOC dataset under the same format as
        the rest of the datasets

        Arguments:
        ----------
        torchvision_pascal_data: torchvision.datasets.VOCSegmentation

        Returns:
        --------
        generic_pascal_data: dict
            Dict where the keys are the ids of the samples and the values are
            also dictionaries with two keys: 'Data' (image) and 'Label' (segmentation mask)
    """
    generic_pascal_data = {}
    for id_current_sample in range(len(torchvision_pascal_data)):
        generic_pascal_data[id_current_sample] = {
                                                    'Data': torchvision_pascal_data[id_current_sample][0],  # Image
                                                    'Label': torchvision_pascal_data[id_current_sample][1]  # Segmentation mask
                                                }

    return generic_pascal_data

class PascalVOCDatasetWrapper(Dataset):
    """
        Pascal VOC dataset wrapper (for the VOCSegmentation torchvision dataset)

        Argument:
        ---------
        data: dict
            Dict where the keys are the ids of the samples and the values are
            also dictionaries with two keys: 'Data' (image) and 'Label' (segmentation mask)
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
