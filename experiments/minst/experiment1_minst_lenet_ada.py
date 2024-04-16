from src.baseexperiment import BaseExperiment

def run_experiment():

    exper_configs = {
        # Context
        'architecture': 'LeNet5',
        'dataset': 'MINST',

        'trainer_args': {
            'trainer': 'KFoldTrainer',

            # 'clip_gradients': True,
            # 'max_gradients': 3,

            # Kfold
            'k_n': 2,

            # Optimizer
            'optimizer': 'SGD',
            'lr': 0.01,
            'momentum': 0.9,
            # 'weight_decay': 0.00001,
            'epochs': 25,

            # Use 20% of train dataset as validation
            'val_ratio': 0.2,

            # Dataset
            'batch_size': 500,
        },

        # Model params
        'model_args': {
            # Activation Function
            'af_name': 'ADA',
            'af_params': {
            }
        }
    }

    expr = BaseExperiment(exper_configs=exper_configs)
    expr.run_experiment()

if __name__ == '__main__':
    run_experiment()