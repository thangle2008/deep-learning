import argparse
import numpy as np
import importlib

from tools import train as Trainer

import keras
import keras.backend as K

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


parser = argparse.ArgumentParser()

parser.add_argument('--data', dest='data', action='store', 
                        choices=['bird', 'tinyimagenet', 'cifar10', 'car'], 
                        default='tinyimagenet')
parser.add_argument('--model', dest='model', action='store', default='resnet')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--optimize', action='store_true')


SEED = 28


def optimize_params(model, train, val, num_classes, dim=224, num_epochs=100):
    """
    Optimize hyperparameters of a given training model.
    """
    space = {
        'lr': hp.uniform('lr', 0.05, 0.5)
    }

    def objective(params):
        score = Trainer.run(model, train, val, num_classes, dim=dim, 
            num_epochs=num_epochs, opt=params)
        return {'loss': score, 'status': STATUS_OK} 

    trials = Trials()

    best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)
    return best


if __name__ == '__main__':
    args = parser.parse_args()

    data_module = importlib.import_module('data.{}'.format(args.data))

    train, val, metadata = data_module.get_data_gen()

    opt = None
    if hasattr(data_module, 'get_optimizer'):
        opt = data_module.get_optimizer()

    Trainer.run(args.model, train, val, metadata, opt=opt, num_epochs=args.epochs)