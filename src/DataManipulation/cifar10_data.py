import torch
import torchvision
from torch.utils.data import Dataset


def put_CIFAR10_data_generic_form(torchvision_cifar10_data):
    """
        Put the data of a torchvision CIFAR10 dataset under the same format as
        the rest of the datasets (HITS, ECG, ESR)

        Arguments:
        ----------
        torchvision_cifar10_data: torchvision.datasets.CIFAR10

        Returns:
        --------
        generic_CIFAR10_data: dict
            Dict where the keys are the ids of the samples and the values are
            also dictionaries with two keys: 'Data' and 'Label'
    """
    generic_CIFAR10_data = {}
    for id_current_sample in range(len(torchvision_cifar10_data)):
        generic_CIFAR10_data[id_current_sample] = {
            'Data': torchvision_cifar10_data[id_current_sample][0],
            'Label': torchvision_cifar10_data[id_current_sample][1]
        }

    return generic_CIFAR10_data

class CIFAR10DatasetWrapper(Dataset):
    """
        CIFAR10 dataset wrapper (for the CIFAR10 torchvision dataset)

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
        print("Sample shape:", sample_data.shape)  # Check shape here
        return sample_data, label
