from src.baseexperiment import BaseExperiment

def run_experiment():

    exper_configs = {
        # Context
        'architecture': 'SimpleNet3D',
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