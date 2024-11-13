import torch
import torchvision
from torch.utils.data import Dataset

def put_CIFAR100_data_generic_form(torchvision_cifar100_data):
    """
        Put the data of a torchvision CIFAR100 dataset under the same format as
        the rest of the datasets (HITS, ECG, ESR)

        Arguments:
        ----------
        torchvision_cifar100_data: torchvision.datasets.CIFAR100

        Returns:
        --------
        generic_CIFAR100_data: dict
            Dict where the keys are the ids of the samples and the values are
            also dictionaries with two keys: 'Data' and 'Label'
    """
    generic_CIFAR100_data = {}
    for id_current_sample in range(len(torchvision_cifar100_data)):
        generic_CIFAR100_data[id_current_sample] = {
            'Data': torchvision_cifar100_data[id_current_sample][0],
            'Label': torchvision_cifar100_data[id_current_sample][1]
        }

    return generic_CIFAR100_data

class CIFAR100DatasetWrapper(Dataset):
    """
        CIFAR100 dataset wrapper (for the CIFAR100 torchvision dataset)

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
