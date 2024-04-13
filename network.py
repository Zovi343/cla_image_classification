# STUDENT's UCO: 000000

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.

import torch.nn as nn
import torch.nn.functional as F


class ModelExample(nn.Module):
    def __init__(self, num_classes):
        super(ModelExample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 64 * 64, 1024)  # Assuming the input image size is 256x256
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input shape (batch_size, 3, 256, 256)
        x = self.pool(F.relu(self.conv1(x)))  # Output shape (batch_size, 32, 128, 128)
        x = self.pool(F.relu(self.conv2(x)))  # Output shape (batch_size, 64, 64, 64)
        x = x.view(-1, 64 * 64 * 64)  # Flatten the output for the dense layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
