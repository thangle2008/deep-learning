import argparse
import importlib

import keras
from tools import trainer, tester

from models.resnet import ResnetBuilder
from models import vgg16


parser = argparse.ArgumentParser()

# for training
parser.add_argument('--data', dest='data', action='store',
                    choices=['bird', 'tinyimagenet', 'cifar10', 'car'])
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


if __name__ == '__main__':
    args = parser.parse_args()

    data_module = importlib.import_module('data.{}'.format(args.data))

    # extract train dimension and number of classes
    dim = data_module.TRAIN_DIM
    num_classes = data_module.NUM_CLASSES

    if args.evaluate is not None:
        test_gen = data_module.get_test_gen(args.evaluate, args.ten_crop)
        tester.evaluate(args.classifier, test_gen, args.ten_crop,
                        output_dir=args.output_false)

    else:
        train, val = data_module.get_data_gen()

        # load model
        print "Load model..."

        model = None
        if args.model == 'resnet':
            model = ResnetBuilder.build_resnet(
                (3, dim, dim), num_classes,
                depth=args.depth, base_filters=args.filters,
                downsampling_top=args.pooling, shortcut_option=args.shortcut)
        elif args.model == 'vgg16':
            weights = 'imagenet' if args.pretrained else None
            model = vgg16.build_model((dim, dim, 3), num_classes,
                                      weights=weights)

        # config optimizer
        if args.algo == 'sgd':
            optimizer = keras.optimizers.SGD(
                lr=args.lr, nesterov=True, momentum=0.9)
        else:
            optimizer = keras.optimizers.Adam()

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        trainer.run(model, train, val, num_epochs=args.epochs)