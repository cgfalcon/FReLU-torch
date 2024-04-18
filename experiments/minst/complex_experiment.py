from src.baseexperiment import BaseExperiment

def run_experiment():

    exper_configs = [
        {
        # Context
        'architecture': 'VGG11Net',
        'dataset': 'MINST',

        'trainer_args': {
            'trainer': 'KFoldTrainer',

            # 'clip_gradients': True,
            # 'max_gradients': 3,

            # Kfold
            'k_n': 2,

            # Optimizer
            'optimizer': 'ADAM',
            'lr': 0.0001,
            'momentum': 0.9,
            'weight_decay': 0.00001,
            # 'optimizer': 'SGD',
            # 'lr': 0.01,
            # 'momentum': 0.9,
            # 'weight_decay': 0.00001,
            'epochs': 20,

            # Use 20% of train dataset as validation
            'val_ratio': 0.2,

            # Dataset
            'batch_size': 500,
        },

        # Model params
        'model_args': {
            # Activation Function
            'af_name': 'GCU',
            'af_params': {
            }
        }
    }
    ]

    for epr in exper_configs:
        model_args = epr['model_args']
        af = model_args['af_name']
        print(f'===== [{af}] ===')
        expr = BaseExperiment(exper_configs=epr)
        try:
            expr.run_experiment()
        except Exception as e:
            print(f'Failed: {af}, {e}')

if __name__ == '__main__':
    run_experiment()