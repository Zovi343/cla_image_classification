# STUDENT's UCO: 000000

# Description:
# This file should contain custom dataset class. The class should subclass the torch.utils.data.Dataset.
import torch
from torch.utils.data import Dataset

class SampleDataset(Dataset):

    def __init__(self):
        pass


    def __len__(self):
        return 1

    def __getitem__(self, idx):

        return torch.zeros((256,256))


