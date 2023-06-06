import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from skimage.measure import shannon_entropy


def load_data(
        *,
        hr_data_dir,
        lr_data_dir,
        other_data_dir,
        deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param lr_data_dir:
    :param other_data_dir:
    :param hr_data_dir:
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param deterministic: if True, yield results in a deterministic order.
    """
    return BraTSMRI(hr_data_dir, lr_data_dir, other_data_dir)


class BraTSMRI(Dataset):
    def __init__(self, hr_data_name, lr_data_name, other_data_name):
        self.hr_data, self.lr_data, self.other_data = np.load(hr_data_name, mmap_mode="r")[:, 40:60 ], \
                                                      np.load(lr_data_name, mmap_mode="r")[:, 40:60], \
                                                      np.load(other_data_name, mmap_mode="r")[:, 40:60]

        num_subject, num_slice, h, w = self.hr_data.shape
        self.hr_data = self.hr_data.reshape(num_subject * num_slice, h, w)
        self.lr_data = self.lr_data.reshape(num_subject * num_slice, h, w)
        self.other_data = self.other_data.reshape(num_subject * num_slice, h, w)

        data_dict = {}
        for s in range(len(self.hr_data)):
            entropy = np.round(shannon_entropy(self.hr_data[s]))
            if entropy in data_dict:
                data_dict[entropy].append(s)
            else:
                data_dict[entropy] = [s]

        self.hr_data = torch.from_numpy(self.hr_data).float()
        self.lr_data = torch.from_numpy(self.lr_data).float()
        self.other_data = torch.from_numpy(self.other_data).float()

        self.hr_data = torch.unsqueeze(self.hr_data, 1)
        self.lr_data = torch.unsqueeze(self.lr_data, 1)
        self.other_data = torch.unsqueeze(self.other_data, 1)
        self.data_dict = data_dict
        print(self.hr_data.shape, self.lr_data.shape, self.other_data.shape)

    def __len__(self):
        return self.hr_data.shape[0]

    def __getitem__(self, index):
        return self.hr_data[index], self.lr_data[index], self.other_data[index]
