import torch
import torchvision
from torch.utils.data import Dataset

def put_STL10_data_generic_form(torchvision_stl10_data):
    """
    Put the data of a torchvision STL10 dataset under the same format as
    the rest of the datasets (HITS, ECG, ESR)

    Arguments:
    ----------
    torchvision_stl10_data: torchvision.datasets.STL10

    Returns:
    --------
    generic_STL10_data: dict
        Dict where the keys are the ids of the samples and the values are
        also dictionaries with two keys: 'Data' and 'Label'
    """
    generic_STL10_data = {}
    for id_current_sample in range(len(torchvision_stl10_data)):
        generic_STL10_data[id_current_sample] = {
            'Data': torchvision_stl10_data[id_current_sample][0],
            'Label': torchvision_stl10_data[id_current_sample][1]
        }

    return generic_STL10_data

class STL10DatasetWrapper(Dataset):
    """
    STL10 dataset wrapper (for the STL10 torchvision dataset)

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
