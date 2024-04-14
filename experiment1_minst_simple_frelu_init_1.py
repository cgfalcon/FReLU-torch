
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
        'model': 'SimpleFReLUModel',
        'dataset': 'MINST',

        # Optimizer
        'lr': 0.1,
        'momentum': 0.9,
        'epochs': 10,

        # Dataset
        'batch_size': 500,

        # Model Params
        'model_args': {
            'frelu_init': 1
        }
    }

    expr = BaseExperiment(exper_configs=exper_configs)
    expr.run_experiment()


if __name__ == '__main__':
    run_experiment()