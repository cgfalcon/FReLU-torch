import torch

from src.baseexperiment import BaseExperiment, SensitiveExperiment

def run_experiment():

    exper_configs = {
        # Context
        'architecture': 'CompactNet3D',
        'dataset': 'CIFAR10',

        # sensitive params - lr
        'sensitive_param': 'lr',
        'sensitive_param_range': torch.logspace(start=-3, end=-5, steps=3),

        # sensitive params - alpha
        # 'sensitive_param': 'alpha',
        # 'sensitive_param_range': torch.arange(0.3, 0.6, 0.1),

        'trainer_args': {
            'trainer': 'BasicTrainer',

            # 'clip_gradients': True,
            # 'max_gradients': 3,

            # Kfold
            'k_n': 2,

            # Optimizer
            'optimizer': 'ADAM',
            'lr': 0.0001,
            'momentum': 0.9,
            'weight_decay': 0.00001,
            'epochs': 50,

            # Use 20% of train dataset as validation
            'val_ratio': 0.2,

            # Dataset
            'batch_size': 500,
        },

        # Model params
        'model_args': {
            # Activation Function
            'af_name': 'ShiftedQuadraticUnit',
            'af_params': {
            }
        }
    }

    expr = SensitiveExperiment(exper_configs=exper_configs)
    expr.run_experiment()

if __name__ == '__main__':
    run_experiment()