import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
import random
import torch.utils.data as data
import os
from tqdm import tqdm
from glob import glob
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


# These functions may be helpful for you :)
def data_crop(data_raw, obj_len):
    data_len = np.size(data_raw)
    a = random.randint(0, data_len - obj_len)
    b = a + obj_len
    data_cropped = np.array(data_raw[:, a:b])
    return data_cropped


def data_pad(data_raw, obj_len):
    data_len = np.size(data_raw)
    pad_len = obj_len - data_len
    b = np.zeros((1, pad_len))
    data_padded = np.hstack((data_raw, b))
    return data_padded



class EcgDataset(data.Dataset):
    def __init__(self, root, data_len, transform=None,
                 target_transform=None):  # transform=x_transform, target_transform=y_transform
        """
        root: the directory of the data
        data_len: Unknown parameters, but I think it is helpful for you :)
        transform: pre-process for data
        target_transform: target_transform for label
        """

        self.ecgs = []
        self.ecgs = sorted(list(glob(os.path.join(root, '*.mat'))))
        # print(self.ecgs)
        self.transform = transform
        self.target_transform = target_transform
        self.data_len = data_len

    def __getitem__(self, index):
        val_dict_path = self.ecgs[index]
        val_dict = scio.loadmat(val_dict_path)
        ecg_x = val_dict['value']
        ecg_x_len = np.size(ecg_x)

        # TODO: Note that there may need some pre-process for data with different sizes
        # Write your code here
        # if self.data_len < ecg_x_len:
        #     ecg_x = data_crop(ecg_x, self.data_len)
        # else:
        #     ecg_x = data_pad(ecg_x, self.data_len)
        ecg_y = val_dict['label']

        # if self.transform is not None:
        #     ecg_x = self.transform(ecg_x)
        #     ecg_x = ecg_x.squeeze(dim=1).type(torch.FloatTensor)
        # if self.target_transform is not None:
        #     ecg_y = self.target_transform(ecg_y)
        #
        #     ecg_y = ecg_y.squeeze(-1).type(torch.FloatTensor)
        return ecg_x.squeeze(), ecg_y.squeeze(-1)

    def __len__(self):
        return len(self.ecgs)


class TrainDataSet(data.Dataset):
    def __init__(self, traindatapath, data_len, transform=None,
                 target_transform=None):  # transform=x_transform, target_transform=y_transform
        """
        root: the directory of the data
        data_len: Unknown parameters, but I think it is helpful for you :)
        transform: pre-process for data
        target_transform: target_transform for label
        """

        self.ecgs = []
        self.ecgs = traindatapath
        # print(self.ecgs)
        self.transform = transform
        self.target_transform = target_transform
        self.data_len = data_len

    def __getitem__(self, index):
        val_dict_path = self.ecgs[index]
        val_dict = scio.loadmat(val_dict_path)
        ecg_x = val_dict['value']
        ecg_x_len = np.size(ecg_x)

        # TODO: Note that there may need some pre-process for data with different sizes
        # Write your code here
        if self.data_len < ecg_x_len:
            ecg_x = data_crop(ecg_x, self.data_len)
        else:
            ecg_x = data_pad(ecg_x, self.data_len)
        ecg_y = val_dict['label']

        if self.transform is not None:
            ecg_x = self.transform(ecg_x)
            ecg_x = ecg_x.squeeze(dim=1).type(torch.FloatTensor)
        if self.target_transform is not None:
            ecg_y = self.target_transform(ecg_y)

            ecg_y = torch.flatten(ecg_y).type(torch.int64)[0]
        return ecg_x, F.one_hot(ecg_y,num_classes=2).type(torch.FloatTensor)

    def __len__(self):
        return len(self.ecgs)

class TestDataSet(data.Dataset):
    def __init__(self, testdatapath, data_len, transform=None,
                 target_transform=None):  # transform=x_transform, target_transform=y_transform
        """
        root: the directory of the data
        data_len: Unknown parameters, but I think it is helpful for you :)
        transform: pre-process for data
        target_transform: target_transform for label
        """

        self.ecgs = []
        self.ecgs = testdatapath
        # print(self.ecgs)
        self.transform = transform
        self.target_transform = target_transform
        self.data_len = data_len

    def __getitem__(self, index):
        val_dict_path = self.ecgs[index]
        val_dict = scio.loadmat(val_dict_path)
        ecg_x = val_dict['value']
        ecg_x_len = np.size(ecg_x)

        # TODO: Note that there may need some pre-process for data with different sizes
        # Write your code here
        if self.data_len < ecg_x_len:
            ecg_x = data_crop(ecg_x, self.data_len)
        else:
            ecg_x = data_pad(ecg_x, self.data_len)
        ecg_y = val_dict['label']

        if self.transform is not None:
            ecg_x = self.transform(ecg_x)
            ecg_x = ecg_x.squeeze(dim=1).type(torch.FloatTensor)
        if self.target_transform is not None:
            ecg_y = self.target_transform(ecg_y)

            ecg_y = torch.flatten(ecg_y).type(torch.int64)[0]
        return ecg_x, F.one_hot(ecg_y, num_classes=2).type(torch.FloatTensor)

    def __len__(self):
        return len(self.ecgs)
