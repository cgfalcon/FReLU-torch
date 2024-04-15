
import torch
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.model_selection import KFold
import numpy as np

import wandb

from src.models import *

architectures = {
    'SimpleNet': SimpleNet,
    'VGG11Net': VGG11Net,
    'SimpleNet3D': SimpleNet3D,
    'VGG11Net3D': VGG11Net3D,
    'SmallNet': SmallNet,
    'SmallNet3D': SmallNet3D,
    'OsciAFNet3D': OsciAFNet3D,
    'LeNet5': LeNet5,
    'LeNet53D': LeNet53D
}


class BasicTrainer(object):

    def __init__(self, arch_name, device, trainer_args):
        super().__init__()

        if arch_name not in architectures.keys():
            raise ValueError('Unknown model: {}'.format(arch_name))

        self.arch_name = arch_name

        # Examine key properties of train args
        self._examine_params(trainer_args)

        self.trainer_args = trainer_args
        self.optimizer_name = self.trainer_args['optimizer']

        self.max_epoc = int(self.trainer_args['epochs'])
        self.lr = float(self.trainer_args['lr'])

        self.batch_size = int(self.trainer_args['batch_size'])

        momentum = self.trainer_args['momentum'] if 'momentum' in self.trainer_args else 0.9
        weight_decay = self.trainer_args['weight_decay'] if 'weight_decay' in self.trainer_args else 0.0

        self.momentum = momentum
        self.weight_decay = weight_decay

        self.clip_gradients = False if 'clip_gradients' not in self.trainer_args else self.trainer_args['clip_gradients']
        self.max_gradients = 5 if 'max_gradients' not in self.trainer_args else self.trainer_args['max_gradients']

        # Initialize loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()

        print(f'Trainer configs: \n')
        print(f'  arch_name: {self.arch_name}')
        print(f'  optimizer_name: {self.optimizer_name}')
        print(f'  max_epoc: {self.max_epoc}')
        print(f'  lr: {self.lr}')
        print(f'  momentum: {self.momentum}')
        print(f'  weight_decay: {self.weight_decay}')

        self.device = device
        self.train_summary = {}
        self.test_summary = {}

    def _examine_params(self, train_args):
        if 'batch_size' not in train_args:
            raise ValueError('Missing config: batch_size.')

        if 'val_ratio' not in train_args:
            raise ValueError('Missing config: val_ratio.')

        if 'optimizer' not in train_args:
            raise ValueError('Missing config: optimizer.')

        if 'lr' not in train_args:
            raise ValueError('Missing config: lr')

    def train_model(self, model_args, train_dastaset):
        """Training loop for any model"""

        # Split dataset into train and validation
        val_ratio = float(self.trainer_args['val_ratio'])

        val_size = int(len(train_dastaset) * val_ratio)
        train_size = len(train_dastaset) - val_size

        train_subset, val_subset = random_split(train_dastaset, [train_size, val_size])
        print(f'Train size ({100 * (1 - val_ratio)}%): {len(train_subset)}, Val size ({100 * val_ratio}%): {len(val_subset)}')

        train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

        print(f'\tTraining set {len(train_dataloader)} has instances')
        print(f'\tTest_loader set {len(val_dataloader)} has instances')

        model = architectures[self.arch_name](**model_args)
        print(f'Model {self.arch_name} loaded, {model}')
        # Move model to device
        model.to(self.device)

        # reset train summary
        self.train_summary = {}
        self.train_losses = []
        self.val_losses = []

        self.train_acc = []
        self.val_acc = []

        if (self.optimizer_name == 'SGD'):
            optimizer = torch.optim.SGD(model.parameters(), momentum=self.momentum, lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.max_epoc):
            print(f'[Epoch {epoch}]')
            avg_train_acc, avg_train_loss, avg_val_acc, avg_val_loss = self.train_one_epoch(self.loss_fn, model,
                                                                                            optimizer, train_dataloader,
                                                                                            val_dataloader)

            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)

            self.train_acc.append(avg_train_acc)
            self.val_acc.append(avg_val_acc)

            wandb.log({'training/train_loss': avg_train_loss,
                       'training/train_acc': avg_train_acc,
                       'training/val_loss': avg_val_loss,
                       'training/val_acc': avg_val_acc, 'epoch': epoch})

            print(
                f'\tLoss => Train loss: {avg_train_loss:.2f}, Test loss: {avg_val_loss:.2f}, Train Acc: {avg_train_acc:.2f}, Test Acc: {avg_val_acc:.2f}')

        self.model = model
        self.train_summary = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_acc': self.train_acc,
            'val_acc': self.val_acc
        }
        return self.train_summary

    # def start_train(self, loss_fn, optimizer, max_epoc, model, train_dataloader, val_dataloader):
    #     for epoch in range(max_epoc):
    #         print(f'[Epoch {epoch}]')
    #         avg_train_acc, avg_train_loss, avg_val_acc, avg_val_loss = self.train_one_epoch(loss_fn, model,
    #                                                                                         optimizer, train_dataloader,
    #                                                                                         val_dataloader)
    #
    #         self.train_losses.append(avg_train_loss)
    #         self.val_losses.append(avg_val_loss)
    #
    #         self.train_acc.append(avg_train_acc)
    #         self.val_acc.append(avg_val_acc)
    #
    #         wandb.log({'train_loss': avg_train_loss,
    #                    'train_acc': avg_train_acc,
    #                    'val_loss': avg_val_loss,
    #                    'val_acc': avg_val_acc})
    #
    #         print(
    #             f'\tLoss => Train loss: {avg_train_loss:.2f}, Test loss: {avg_val_loss:.2f}, Train Acc: {avg_train_acc:.2f}, Test Acc: {avg_val_acc:.2f}')

    def train_one_epoch(self, loss_fn, model, optimizer, train_dataloader, val_dataloader):
        model.train(True)
        avg_train_loss = 0.
        avg_train_acc = 0.0
        for b, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # clear gradient
            optimizer.zero_grad()

            outputs = model(inputs)

            # loss
            loss = loss_fn(outputs, labels)
            loss.backward()

            if self.clip_gradients:
                nn.utils.clip_grad_norm_(model.parameters(), self.max_gradients)

            # update parameters
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).float().sum()
            acc = correct / inputs.size(0)

            avg_train_acc += acc.cpu()
            avg_train_loss += loss.item()
            if b % 10 == 0:
                print('\tBatch {} loss: {}'.format(b + 1, loss.item()))
        avg_train_loss = avg_train_loss / len(train_dataloader)  # avg loss per batch
        avg_train_acc = avg_train_acc / len(train_dataloader)
        # Start evaluation
        avg_val_loss = 0.
        avg_val_acc = 0.0
        model.eval()
        with torch.no_grad():
            for b, data in enumerate(val_dataloader):
                test_inputs, test_labels = data
                test_inputs = test_inputs.to(self.device)
                test_labels = test_labels.to(self.device)

                test_outputs = model(test_inputs)

                # Loss
                test_loss = loss_fn(test_outputs, test_labels)

                # Accuracy
                _, predicted = torch.max(test_outputs, 1)
                correct = (predicted == test_labels).float().sum()
                acc = correct / test_inputs.size(0)

                avg_val_acc += acc.cpu()
                avg_val_loss += test_loss.item()
        avg_val_loss = avg_val_loss / len(val_dataloader)
        avg_val_acc = avg_val_acc / len(val_dataloader)
        return avg_train_acc, avg_train_loss, avg_val_acc, avg_val_loss

    def test_model(self, test_dataset):
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        avg_val_loss = 0.
        avg_val_acc = 0.0

        self.test_summary = {}

        self.model.eval()
        with torch.no_grad():
            for b, data in enumerate(test_dataloader):
                test_inputs, test_labels = data
                test_inputs = test_inputs.to(self.device)
                test_labels = test_labels.to(self.device)

                test_outputs = self.model(test_inputs)

                # Loss
                test_loss = self.loss_fn(test_outputs, test_labels)

                # Accuracy
                _, predicted = torch.max(test_outputs, 1)
                correct = (predicted == test_labels).float().sum()
                acc = correct / test_inputs.size(0)

                avg_val_acc += acc.cpu()
                avg_val_loss += test_loss.item()
        avg_val_loss = avg_val_loss / len(test_dataloader)
        avg_val_acc = avg_val_acc / len(test_dataloader)

        print(f'Test Loss: {avg_val_loss:.2f}, Accuracy: {avg_val_acc:}')
        self.test_summary = {
            'test_loss': avg_val_loss,
            'test_acc': avg_val_acc,
        }

        wandb.log({'testing/test_loss': avg_val_loss,
                   'testing/test_acc': avg_val_acc, 'epoch': 0})

        return self.test_summary

    def get_train_summary(self):
        return self.train_summary

    def get_model(self):
        return self.model



class KFoldTrainer(BasicTrainer):

    def __init__(self, arch_name, device, trainer_args):
        super().__init__(arch_name, device, trainer_args)

        if 'k_n' not in trainer_args:
            raise ValueError('Missing fold num: "k_n"')

        self.k_n = trainer_args['k_n']

    def train_model(self, model_args, train_dastaset):
        kfold = KFold(n_splits=self.k_n, shuffle=True)

        total_train_losses = []
        total_val_losses = []

        total_train_acc = []
        total_val_acc = []

        # Loop over each fold
        for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(train_dastaset)))):
            fold_n = fold + 1

            # Define wandb metrics
            wandb.define_metric(f'Fold-{fold_n}/train_loss', step_metric="epoch")
            wandb.define_metric(f'Fold-{fold_n}/train_acc', step_metric="epoch")
            wandb.define_metric(f'Fold-{fold_n}/val_loss', step_metric="epoch")
            wandb.define_metric(f'Fold-{fold_n}/val_acc', step_metric="epoch")

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

            train_dataloader = torch.utils.data.DataLoader(train_dastaset, batch_size=self.batch_size, sampler=train_subsampler)
            val_dataloader = torch.utils.data.DataLoader(train_dastaset, batch_size=self.batch_size, sampler=val_subsampler)

            print(f'Fold: {fold_n}, Train Batches: {len(train_dataloader)}, Val Batches: {len(val_dataloader)}')

            model = architectures[self.arch_name](**model_args)
            model.to(self.device)
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), momentum=self.momentum, lr=self.lr, weight_decay=self.weight_decay)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            # reset train summary
            self.train_summary = {}

            train_losses = []
            val_losses = []

            train_acc = []
            val_acc = []

            for epoch in range(self.max_epoc):
                print(f'Fold {fold_n} - [Epoch {epoch}]')
                avg_train_acc, avg_train_loss, avg_val_acc, avg_val_loss = self.train_one_epoch(self.loss_fn, model,
                                                                                        optimizer, train_dataloader,
                                                                                        val_dataloader)

                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)

                train_acc.append(avg_train_acc)
                val_acc.append(avg_val_acc)

                wandb.log({f'Fold-{fold_n}/train_loss': avg_train_loss,
                   f'Fold-{fold_n}/train_acc': avg_train_acc,
                   f'Fold-{fold_n}/val_loss': avg_val_loss,
                   f'Fold-{fold_n}/val_acc': avg_val_acc, 'epoch': epoch})

                print(
                    f'\tFold {fold_n} Loss => Train loss: {avg_train_loss:.2f}, Test loss: {avg_val_loss:.2f}, Train Acc: {avg_train_acc:.2f}, Test Acc: {avg_val_acc:.2f}')
            self.model = model
            fold_summary = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_acc': train_acc,
                'val_acc': val_acc
            }

            total_train_losses.append(train_losses)
            total_val_losses.append(val_losses)
            total_train_acc.append(train_acc)
            total_val_acc.append(val_acc)
            # save summary for every fold
            self.train_summary[fold_n] = fold_summary

        fold_train_losses = torch.mean(torch.tensor(total_train_losses), 0)
        fold_val_losses = torch.mean(torch.tensor(total_val_losses), 0)
        fold_train_acc = torch.mean(torch.tensor(total_train_acc), 0)
        fold_val_acc = torch.mean(torch.tensor(total_val_acc), 0)

        log_idx = 0
        for train_loss, val_loss, train_acc, val_acc in zip(fold_train_losses, fold_val_losses, fold_train_acc, fold_val_acc):
            wandb.log({f'training/train_loss': train_loss,
                       f'training/train_acc': train_acc,
                       f'training/val_loss': val_loss,
                       f'training/val_acc': val_acc, 'epoch': log_idx})
            log_idx += 1


        return self.train_summary