import argparse
import numpy as np
import importlib

from tools import train as Trainer
from tools import test as Tester

import keras
import keras.backend as K

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


parser = argparse.ArgumentParser()

# for training
parser.add_argument('--data', dest='data', action='store', 
                        choices=['bird', 'tinyimagenet', 'cifar10', 'car'], 
                        default='tinyimagenet')
parser.add_argument('--model', dest='model', action='store', default='resnet')
parser.add_argument('--algo', dest='algo', action='store', default='adam')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--optimize', action='store_true')

# for resnet
parser.add_argument('--depth', type=int, default=18)

# for testing
parser.add_argument('--evaluate', action='store', choices=['train', 'val', 'test'])
parser.add_argument('--classifier', action='store')


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


    if args.algo == 'sgd':
        opt = {
            'algo': args.algo,
            'params': {
                'nesterov': True,
                'lr': args.lr,
                'momentum': 0.9
            }
        }
    else:
        opt = {'algo': 'adam'}


    if args.evaluate is not None:
        test_gen = data_module.get_test_gen(args.evaluate)
        Tester.evaluate(args.classifier, test_gen)
    else:
        train, val = data_module.get_data_gen()

        Trainer.run(args.model, train, val, opt=opt, num_epochs=args.epochs, 
                    depth=args.depth)