
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

        tr_key = 'BT'
        trainer_name = self.trainer_args['trainer']
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

        self._do_experiment(trainer, model_args)
        # except Exception as e:
        #     print(f'Experiment failed with exception, \n {e}')
        #     wandb.finish()

        wandb.finish()

    def _do_experiment(self,  trainer, model_args):
        # try:
        trainer.train_model(model_args, self.train)

        return trainer.test_model(self.test)


class SensitiveExperiment(BaseExperiment):

    def __init__(self, exper_configs):
        super().__init__(exper_configs)

        self._check_sensitive_params(exper_configs)
        self.sensitive_param = exper_configs['sensitive_param']
        self.sensitive_param_range = exper_configs['sensitive_param_range']

        model_args = self.exper_configs['model_args']
        self.af_name = model_args['af_name']

    def _check_sensitive_params(self, exper_configs):
        if 'sensitive_param' not in exper_configs:
            raise ValueError('Missing param: sensitive_param')

        if 'sensitive_param_range' not in exper_configs:
            raise ValueError('Missing param: sensitive_param_range')

        sensitive_param_range = exper_configs['sensitive_param_range']
        if not isinstance(sensitive_param_range, torch.Tensor):
            raise ValueError('sensitive_param_range should be a torch.Tensor')

    def run_experiment(self):
        wandb.login()

        param_acc = {}

        for pr in self.sensitive_param_range:
            print(f'Sensitive param {self.sensitive_param} with value: {pr}')

            model_args = self.exper_configs['model_args']
            trainer_args = self.exper_configs['trainer_args']

            # Update config
            if self.sensitive_param == 'lr':
                trainer_args['lr'] = pr
                print(f'Trainer updated configs:')
                print(trainer_args)
            else:
                af_args = model_args['af_params']
                af_args[self.sensitive_param] = pr
                print(f'Model updated configs:')
                print(model_args)

            expr_key = datetime.now().strftime('%Y%m%d%H%M')

            tr_key = 'BT'
            trainer_name = self.trainer_args['trainer']
            if trainer_name == 'KFoldTrainer':
                trainer = KFoldTrainer(self.arch_name, self.device, self.trainer_args)
                tr_key = '[KF]'
            else:
                trainer = BasicTrainer(self.arch_name, self.device, self.trainer_args)
                tr_key = '[BT]'

            expr_name = f"{self.sensitive_param}[{pr:.2f}]-{self.af_name}-{self.dataset_name}-{self.arch_name}-{expr_key}"
            print(f'\tSensitive Experiment[{expr_name}] is running')

            wandb.init(
                # Set the project where this run will be logged
                project='SensiParams',
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=expr_name,
                # Track hyperparameters and run metadata
                config={**self.exper_configs},
                group=f'Sensitive-{self.af_name}-{self.sensitive_param}'
            )

            wandb.define_metric("epoch")
            wandb.define_metric('training/train_loss', step_metric="epoch")
            wandb.define_metric('training/train_acc', step_metric="epoch")
            wandb.define_metric('training/val_loss', step_metric="epoch")
            wandb.define_metric('training/val_acc', step_metric="epoch")
            wandb.define_metric('testing/test_loss', step_metric="epoch")
            wandb.define_metric('testing/test_acc', step_metric="epoch")

            try:
                pr_test_acc = self._do_experiment(trainer, model_args)
                param_acc[pr] = pr_test_acc
                wandb.finish()
            except Exception as e:
                print(f'Experiment failed with exception, \n {e}')
                wandb.finish()

        expr_key = datetime.now().strftime('%Y%m%d%H%M')
        print(f'Sensitive-{self.af_name}-{self.sensitive_param}-summary-{expr_key}')
        print(param_acc)
        #
        #
        # # Report Summary
        # wandb.init(
        #     # Set the project where this run will be logged
        #     project='SensiParams',
        #     # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        #     name=f'Sensitive-{self.af_name}-{self.sensitive_param}-summary-{expr_key}',
        #     # Track hyperparameters and run metadata
        #     config={**self.exper_configs},
        #     group=f'Sensitive-{self.af_name}-{self.sensitive_param}'
        # )
        #
        # wandb.define_metric(self.sensitive_param)
        # wandb.define_metric('sensiparam/test_acc', step_metric=self.sensitive_param)



