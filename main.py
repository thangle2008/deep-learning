import argparse
import importlib

from tools import trainer
from tools import tester

parser = argparse.ArgumentParser()

# for training
parser.add_argument('--data', dest='data', action='store',
                    choices=['bird', 'tinyimagenet', 'cifar10', 'car'])
parser.add_argument('--classes', dest='classes', type=int)
parser.add_argument('--model', dest='model', action='store', default='resnet')
parser.add_argument('--algo', dest='algo', action='store', default='sgd')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--optimize', action='store_true')
parser.add_argument('--pretrained', action='store_true')

# for Resnet
parser.add_argument('--depth', type=int, default=18)
parser.add_argument('--filters', type=int, default=64)
parser.add_argument('--pooling', action='store_true')
parser.add_argument('--shortcut', action='store', choices=['A', 'B'], default='B')

# for testing
parser.add_argument('--evaluate', action='store',
                    choices=['train', 'val', 'test'])
parser.add_argument('--output_false', action='store')
parser.add_argument('--classifier', action='store')
parser.add_argument('--ten_crop', action='store_true')


SEED = 28


def optimize_params(model, train, val, num_outputs, dim, num_epochs=10,
                    **kwargs):
    """
    Optimize hyperparameters of a given training model.
    """

    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

    space = {
        'lr': hp.choice('lr', [0.001, 0.01, 0.1]),
        'nesterov': True,
        'momentum': 0.9
    }

    def objective(params):
        opt = {
            'algo': 'sgd',
            'params': params
        }
        score = trainer.run(model, train, val, opt, num_outputs, dim,
                            num_epochs=num_epochs, **kwargs)
        return {'loss': score, 'status': STATUS_OK} 

    trials = Trials()

    best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)
    return best


if __name__ == '__main__':
    args = parser.parse_args()

    data_module = importlib.import_module('data.{}'.format(args.data))

    # config for resnet
    net_config = {
        'depth': args.depth,
        'base_filters': args.filters,
        'shortcut_option': args.shortcut,
        'downsampling_top': args.pooling
    }

    if args.pretrained:
        net_config['pretrained'] = True

    # configure the optimizer
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
        test_gen = data_module.get_test_gen(args.evaluate, args.ten_crop)
        tester.evaluate(args.classifier, test_gen, args.ten_crop,
                        output_dir=args.output_false)

    elif args.optimize:
        train, val = data_module.get_data_gen()
        print optimize_params(args.model, train, val, args.classes,
                              train.output_shape[0], num_epochs=10,
                              **net_config)
    else:
        train, val = data_module.get_data_gen()

        trainer.run(args.model, train, val, opt, args.classes,
                    train.output_shape[0], num_epochs=args.epochs,
                    **net_config)