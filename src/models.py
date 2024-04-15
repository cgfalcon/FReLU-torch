
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.activations import FReLU, AFFactory
import numpy


affactory = AFFactory()

class SimpleNet(nn.Module):
    """Simple Neural Network with 1 input channel"""

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


class SimpleNet3D(nn.Module):

    def __init__(self, af_name, af_params):
        super(SimpleNet3D, self).__init__()
        print(f'SimpleNet3D initialized with params: \n {af_params}')

        A_F = affactory.get_activation(af_name, af_params)

        self.con_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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

class SmallNet(nn.Module):
    """Simple Neural Network with 1 input channel"""

    def __init__(self, af_name, af_params):
        super(SmallNet, self).__init__()
        print(f'SmallNet initialized with params: \n {af_params}')

        A_F = affactory.get_activation(af_name, af_params)

        self.con_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            A_F,
            nn.Dropout(p=0.5),

            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.con_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

class SmallNet3D(nn.Module):
    """Simple Neural Network with 1 input channel"""

    def __init__(self, af_name, af_params):
        super(SmallNet3D, self).__init__()
        print(f'SmallNet3D initialized with params: \n {af_params}')

        A_F = affactory.get_activation(af_name, af_params)

        self.con_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            A_F,
            nn.Dropout(p=0.5),

            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.con_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

class VGG11Net(nn.Module):
    """VGG11Net with 1 input channel"""

    def __init__(self, af_name, af_params):
        super(VGG11Net, self).__init__()

        print(f'VGG11Net initialized with params: \n {af_params}')

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

class VGG11Net3D(nn.Module):
    """VGG11Net with 1 input channel"""

    def __init__(self, af_name, af_params):
        super(VGG11Net3D, self).__init__()

        print(f'VGG11Net3D initialized with params: \n {af_params}')

        A_F = affactory.get_activation(af_name, af_params)

        self.convs_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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



class OsciAFNet3D(nn.Module):
    """VGG11Net with 1 input channel"""

    def __init__(self, af_name, af_params):
        super(OsciAFNet3D, self).__init__()

        print(f'OsciAFNet3D initialized with params: \n {af_params}')

        A_F = affactory.get_activation(af_name, af_params)

        self.convs_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            A_F,

            nn.Conv2d(32, 32, kernel_size=3),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            A_F,

            nn.Conv2d(64, 64, kernel_size=3),
            A_F,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2304, 512),
            A_F,
            nn.Dropout(p=0.5),

            nn.Linear(512, 10),
            A_F
        )

    def forward(self, x):
        x = self.convs_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
