from src.baseexperiment import BaseExperiment

def run_experiment():

    exper_configs = {
        # Context
        'architecture': 'VGG11Net',
        'dataset': 'MINST',

        # Optimizer
        'lr': 0.0001,
        'epochs': 10,
        'optimizer': 'ADAM', # 'SGD' or 'ADAM

        # Dataset
        'batch_size': 500,

        # Model params
        'model_args': {
            # Activation Function
            'af_name': 'ShiftedSincUnit',
            'af_params': {
            }
        }
    }

    expr = BaseExperiment(exper_configs=exper_configs)
    expr.run_experiment()

if __name__ == '__main__':
    run_experiment()