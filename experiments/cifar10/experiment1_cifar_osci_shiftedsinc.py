
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
import torch
from torch.utils.data import Subset
import torch.nn as nn

import wandb

import src
import importlib

from src import *

from baseexperiment import BaseExperiment

def run_experiment():

    exper_configs = {
        # Context
        'architecture': 'OsciAFNet3D',
        'dataset': 'CIFAR10',

        # Optimizer
        'lr': 0.0001,
        'epochs': 50,
        'optimizer': 'ADAM', # 'SGD' or 'ADAM
        'weight_decay': 0.00001,

        # Dataset
        'batch_size': 35,

        # Model params
        'model_args': {
            # Activation Function
            'af_name': 'ShiftedSincUnit',
            'af_params': {
            }
        }
    }

    expr = BaseExperiment(exper_configs=exper_configs)
    expr.run_experiment()

if __name__ == '__main__':
    run_experiment()