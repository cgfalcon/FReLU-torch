
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
import torch
from torch.utils.data import Subset
import torch.nn as nn
import wandb
from datetime import datetime

from src import *

img_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

class BaseExperiment(object):

    def __init__(self, exper_configs):
        super(BaseExperiment, self).__init__()
        print(f'Initializing experiment')

        self.arch_name = exper_configs['architecture']
        self.af_name = exper_configs['model_args']['af_name']
        self.dataset_name = exper_configs['dataset']
        self.exper_configs = exper_configs
        self.subset_size = None if 'subset_size' not in exper_configs else exper_configs['subset_size']

        self.trainer_args = exper_configs['trainer_args']

        # Load dataset
        self.train_loader = None
        self.test_loader = None
        self.init_dataset(self.dataset_name, self.subset_size)

        # Initialize device
        self.find_device()


    def init_dataset(self, dataset_name, subset_size):

        self.dataset_name = dataset_name

        # Download dataset
        print(f'\tLoading {dataset_name}')
        if dataset_name == 'MINST':
            self.train = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=img_transforms)
            self.test = torchvision.datasets.MNIST(root='.', train=False, download=True, transform=img_transforms)

            if subset_size is not None:
                self.train = self.subset_of_minst(self.train, subset_size)
        elif dataset_name == 'CIFAR10':
            self.train = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=img_transforms)
            self.test = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=img_transforms)
        elif dataset_name == 'CIFAR100':
            self.train = torchvision.datasets.CIFAR100(root='.', train=True, download=True, transform=img_transforms)
            self.test = torchvision.datasets.CIFAR100(root='.', train=False, download=True, transform=img_transforms)
        else:
            raise ValueError('Unknown dataset name: {}.'.format(dataset_name))


    def subset_of_minst(self, dataset, subset_size=500):
        """Create subset for given dataset"""
        digit_indices = {digit: [] for digit in range(10)}

        for idx, (_, target) in enumerate(dataset):
            digit_indices[target].append(idx)

        selected_indexes = []
        for digit, indices in digit_indices.items():
            selected_indexes.extend(np.random.choice(indices, subset_size, replace=False))

        sub_set = Subset(dataset, selected_indexes)
        return sub_set

    def find_device(self):
        # Define device
        device = 'cpu'
        # Check if MPS is supported and available
        if torch.backends.mps.is_available():
            print("MPS is available on this device.")
            device = torch.device("mps")  # Use MPS device
            # device = 'cpu'
        else:
            print("MPS not available, using CPU instead.")
            device = torch.device("cpu")  # Fallback to CPU
        self.device = device

    def run_experiment(self, ):
        # login
        wandb.login()

        expr_key = datetime.now().strftime('%Y%m%d%H%M')

        trainer_name = self.trainer_args['trainer']
        tr_key = 'BT'
        if trainer_name == 'KFoldTrainer':
            trainer = KFoldTrainer(self.arch_name, self.device, self.trainer_args)
            tr_key = '[KF]'
        else:
            trainer = BasicTrainer(self.arch_name, self.device, self.trainer_args)
            tr_key = '[BT]'

        expr_name = f"{tr_key}-{self.dataset_name}-{self.arch_name}-{self.af_name}-{expr_key}"
        print(f'\tExperiment[{expr_name}] is running')



        wandb.init(
            # Set the project where this run will be logged
            project="FReLU",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=expr_name,
            # Track hyperparameters and run metadata
            config={**self.exper_configs})

        wandb.define_metric("epoch")
        wandb.define_metric('training/train_loss', step_metric="epoch")
        wandb.define_metric('training/train_acc', step_metric="epoch")
        wandb.define_metric('training/val_loss', step_metric="epoch")
        wandb.define_metric('training/val_acc', step_metric="epoch")
        wandb.define_metric('testing/test_loss', step_metric="epoch")
        wandb.define_metric('testing/test_acc', step_metric="epoch")


        model_args = self.exper_configs['model_args']

        # try:
        trainer.train_model(model_args, self.train)

        trainer.test_model(self.test)

        wandb.finish()
        # except Exception as e:
        #     print(f'Experiment failed with exception, \n {e}')
        #     wandb.finish()