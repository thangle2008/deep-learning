import argparse
import numpy as np
import importlib

from tools import train as Trainer
from tools import test as Tester

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
parser.add_argument('--test', action='store_true')
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

    opt = None
    if hasattr(data_module, 'get_optimizer'):
        opt = data_module.get_optimizer()

    if args.test:
        test_gen, data_size = data_module.get_test_gen()
        Tester.evaluate(args.classifier, test_gen, data_size)
    else:
        train, val, metadata = data_module.get_data_gen()
        Trainer.run(args.model, train, val, metadata, opt=opt, num_epochs=args.epochs)