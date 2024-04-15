
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
        'architecture': 'SmallNet3D',
        'dataset': 'CIFAR10',

        # Optimizer
        'lr': 0.1,
        'momentum': 0.9,
        'epochs': 50,

        # Dataset
        'batch_size': 500,

        # Model params
        'model_args': {
            # Activation Function
            'af_name': 'ReLU',
            'af_params': {
                'inplace': True
            }
        }
    }

    expr = BaseExperiment(exper_configs=exper_configs)
    expr.run_experiment()

if __name__ == '__main__':
    run_experiment()