
import torch
import torch.nn as nn

import wandb

from src.models import *

architectures = {
    'SimpleNet': SimpleNet,
    'VGG11Net': VGG11Net,
    'SimpleNet3D': SimpleNet3D,
    'VGG11Net3D': VGG11Net3D,
    'SmallNet': SmallNet,
    'SmallNet3D': SmallNet3D,
    'OsciAFNet3D': OsciAFNet3D
}


class BasicTrainer(object):

    def __init__(self, arch_name, device, optimizer_name = 'SGD'):
        super().__init__()

        if arch_name not in architectures.keys():
            raise ValueError('Unknown model: {}'.format(arch_name))

        self.arch_name = arch_name
        self.optimizer_name = optimizer_name

        self.device = device
        self.train_summary = {}


    def train_model(self, model_args, train_dataloader, val_dataloader, max_epoc = 20, lr = 0.1, momentum = 0.9, weight_decay = 0):
        """Training loop for any model"""

        model = architectures[self.arch_name](**model_args)
        print(f'Model {self.arch_name} loaded, {model}')
        # Move model to device
        model.to(self.device)

        # reset train summary
        self.train_summary = {}

        train_losses = []
        val_losses = []

        train_acc = []
        val_acc = []

        # Initialize loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        if (self.optimizer_name == 'SGD'):
            optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(max_epoc):
            print(f'[Epoch {epoch}]')

            model.train(True)

            epoch_loss = 0.
            avg_train_loss = 0.

            epoch_acc = 0.0
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

                # update parameters
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).float().sum()
                acc = correct / inputs.size(0)

                epoch_acc += acc.cpu()
                epoch_loss += loss.item()
                if b % 10 == 0:
                    print('\tBatch {} loss: {}'.format(b + 1, loss.item()))

            avg_train_loss = epoch_loss / len(train_dataloader)  # avg loss per batch
            avg_train_acc = epoch_acc / len(train_dataloader)

            # Start evaluation
            running_loss = 0.
            running_acc = 0.0
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

                    running_acc += acc.cpu()
                    running_loss += test_loss.item()
            avg_val_loss = running_loss / len(val_dataloader)
            avg_val_acc = running_acc / len(val_dataloader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            train_acc.append(avg_train_acc)
            val_acc.append(avg_val_acc)

            wandb.log({'train/loss': avg_train_loss,
                       'train/acc': avg_train_acc,
                       'val/loss': avg_val_loss,
                       'val/acc': avg_val_acc})

            print(
                f'\tLoss => Train loss: {avg_train_loss:.2f}, Test loss: {avg_val_loss:.2f}, Train Acc: {avg_train_acc:.2f}, Test Acc: {avg_val_acc:.2f}')

        self.model = model
        self.train_summary = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        return self.train_summary

    def get_train_summary(self):
        return self.train_summary

    def get_model(self):
        return self.model