
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.activations import FReLU
import numpy

class SimpleFReLUModel(nn.Module):

    def __init__(self, frelu_init=1):
        super(SimpleFReLUModel, self).__init__()
        print(f'SimpleFReLU initialized with: frelu_init={frelu_init}')

        self.con_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            FReLU(inplace=True, frelu_init=frelu_init),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            FReLU(inplace=True, frelu_init=frelu_init),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            FReLU(inplace=True, frelu_init=frelu_init)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 64),
            FReLU(inplace=True, frelu_init=frelu_init),

            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.con_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x


class SimpleBasicReLUModel(nn.Module):

    def __init__(self):
        super(SimpleBasicReLUModel, self).__init__()

        self.con_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.con_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x