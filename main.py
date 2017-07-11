import argparse
import importlib

import keras
from tools import trainer, tester

from models.resnet import ResnetBuilder
from models import vgg16


parser = argparse.ArgumentParser()


# data configuration
parser.add_argument('data', action='store',
                    help='the python data configuration file to use')

# training configuration
parser.add_argument('--model', dest='model', action='store',
                    choices=['resnet', 'vgg16'], default='resnet',
                    help='the network model to use')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs')
parser.add_argument('--resume', action='store',
                    help='path to a model file to resume training')
parser.add_argument('--initial_epoch', type=int, default=0,
                    help='the initial epoch to start training')

# optimizer configuration
parser.add_argument_group('optimizer arguments')
parser.add_argument('--algo', dest='algo', action='store', default='sgd',
                    help='the algorithm to use for optimizing the '
                         'loss function')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')

# Resnet-only configuration (read the resnet module to understand more about
# these configurations)
parser.add_argument('--depth', type=int, default=18)
parser.add_argument('--filters', type=int, default=64)
parser.add_argument('--pooling', action='store_true')
parser.add_argument('--shortcut', action='store', choices=['A', 'B'], default='B')

# testing configuration
parser.add_argument('--evaluate', action='store',
                    choices=['train', 'val', 'test'],
                    help='the dataset to use for evaluating')
parser.add_argument('--output_false', action='store',
                    help='path to a folder to output all incorrectly'
                         'classified images')
parser.add_argument('--classifier', action='store',
                    help='path to a trained model')
parser.add_argument('--ten_crop', action='store_true')


SEED = 28


if __name__ == '__main__':
    args = parser.parse_args()

    data_module = importlib.import_module('data.{}'.format(args.data))

    # extract train dimension and number of classes
    dim = data_module.TRAIN_DIM
    num_classes = data_module.NUM_CLASSES

    if args.evaluate is not None:
        test_gen = data_module.get_test_gen(args.evaluate)
        tester.evaluate(args.classifier, test_gen, args.ten_crop,
                        output_dir=args.output_false)

    else:
        train, val = data_module.get_data_gen()

        # load model
        print "Load model..."

        model = None
        if args.resume is not None:
            model = keras.models.load_model(args.resume)
        elif args.model == 'resnet':
            model = ResnetBuilder.build_resnet(
                (3, dim, dim), num_classes,
                depth=args.depth, base_filters=args.filters,
                downsampling_top=args.pooling, shortcut_option=args.shortcut)
        elif args.model == 'vgg16':
            model = vgg16.build_model((dim, dim, 3), num_classes,
                                      weights=None)

        # config optimizer
        if args.resume is None:
            if args.algo == 'sgd':
                optimizer = keras.optimizers.SGD(
                    lr=args.lr, nesterov=True, momentum=0.9)
            else:
                optimizer = keras.optimizers.Adam()

            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        trainer.run(model, train, val, num_epochs=args.epochs,
                    initial_epoch=args.initial_epoch)