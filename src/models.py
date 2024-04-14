
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.activations import FReLU, AFFactory
import numpy


affactory = AFFactory()

class SimpleNet(nn.Module):

    def __init__(self, af_name, af_params):
        super(SimpleNet, self).__init__()
        print(f'SimpleNet initialized with params: \n {af_params}')

        A_F = affactory.get_activation(af_name, af_params)

        self.con_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            A_F
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 64),
            A_F,

            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.con_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x


class VGG11Net(nn.Module):

    def __init__(self, af_name, af_params):
        super(VGG11Net, self).__init__()

        A_F = affactory.get_activation(af_name, af_params)

        self.convs_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            A_F,

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            A_F,

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            A_F,

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            A_F,
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            A_F,
            nn.Dropout(0.5),

            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.convs_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SmallNet(nn.Module):

    def __init__(self, af_name, af_params):
        super(SmallNet, self).__init__()
        print(f'SimpleNet initialized with params: \n {af_params}')

        A_F = affactory.get_activation(af_name, af_params)

        self.con_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            A_F
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 64),
            A_F,

            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.con_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

