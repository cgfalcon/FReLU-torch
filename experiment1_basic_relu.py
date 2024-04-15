from src.baseexperiment import BaseExperiment

def run_experiment():

    exper_configs = {
        # Context
        'model': 'SimpleBasicReLUModel',
        'dataset': 'MINST',

        # Optimizer
        'lr': 0.1,
        'momentum': 0.9,
        'epochs': 10,

        # Dataset
        'batch_size': 500,

        # Model params
        'model_args': { }
    }

    expr = BaseExperiment(exper_configs=exper_configs)
    expr.run_experiment()

if __name__ == '__main__':
    run_experiment()