import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class DNN_power_bands(nn.Module):
    def __init__(self, n_channels: int=128, power_bands: int=5):
        super(DNN_power_bands, self).__init__()
        self.n_channels = n_channels
        self.power_bands = power_bands
        self.fc1 = nn.Linear(n_channels * power_bands, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 2)
        self.dropout = nn.Dropout(p=0.5)

        # Initialize the weights
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')

    def forward(self, x):
        x = x.view(-1, self.n_channels * self.power_bands)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)