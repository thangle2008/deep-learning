import argparse
import json
import numpy as np

from utils.imgloader import load_data, load_imagenet
from tools import train as Trainer

from models import vgg16
from models.resnet import ResnetBuilder

import keras
import keras.backend as K

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


parser = argparse.ArgumentParser()

parser.add_argument('--data', dest='data', action='store', 
                        choices=['caltech101', 'tinyimagenet'], 
                        default='caltech101')
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

    # parse config
    with open('./data/{}.json'.format(args.data)) as data_file:
        dataconf = json.load(data_file)

    datapath = dataconf['data']
    load_dim = dataconf['load_dim']
    dim = dataconf['crop_dim']
    opt = dataconf['opt'] if 'opt' in dataconf else {}

    # load and preprocess data
    print "Load data from", datapath

    if args.data == 'tinyimagenet':
        (X_train, y_train), (X_val, y_val), num_to_name = load_imagenet(datapath)
    else:
        (X_train, y_train), (X_val, y_val), num_to_name = load_data(datapath, 
            new_size=load_dim, p_train=0.8, seed=SEED)

    num_classes = len(num_to_name)

    X_train = K.cast_to_floatx(X_train)
    X_val = K.cast_to_floatx(X_val)

    if 'mean' in dataconf:
        mean = np.asarray(dataconf['mean'], dtype=K.floatx())
        print "Subtracting mean =", mean
        X_train -= mean.reshape(1, 1, 3)
        X_val -= mean.reshape(1, 1, 3)
    
    if 'std' in dataconf:
        std = np.asarray(dataconf['std'], dtype=K.floatx())
        print "Dividing by std =", std
        X_train /= std.reshape(1, 1, 3)
        X_val /= std.reshape(1, 1, 3)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    train, val = (X_train, y_train), (X_val, y_val)

    # load model
    print "Load model..."
    if args.model == 'resnet':
        model = ResnetBuilder.build_resnet_18((3, dim, dim), num_classes)
    elif args.model == 'vgg16':
        model = vgg16.build_model(weights=None, input_shape=(dim, dim, 3), 
            classes=num_classes)

    print "Number of parameters =", model.count_params()
    print "Training samples =", X_train.shape
    print "Number of classes =", num_classes
    
    if args.optimize:
        print "Optimize hyperparameters"
        best = optimize_params(model, train, val, num_classes, 
            dim=dim, num_epochs=10)
        print best

    else:
        Trainer.run(model, train, val, num_classes, 
            dim=dim, num_epochs=args.epochs, opt=opt)