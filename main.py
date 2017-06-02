import argparse
import json
import numpy as np

from tools import train as Trainer
from models.resnet import ResnetBuilder
from utils.imgloader import load_data

import keras
import keras.backend as K

parser = argparse.ArgumentParser()

parser.add_argument('--data', dest='data', action='store', 
                        choices=['caltech101'], default='caltech101')
parser.add_argument('--model', dest='model', action='store', default='resnet')
parser.add_argument('--epochs', type=int, default=100)

SEED = 28


if __name__ == '__main__':
    args = parser.parse_args()

    # parse config
    with open('./data/{}.json'.format(args.data)) as data_file:
        dataconf = json.load(data_file)

    datapath = dataconf['data']
    load_dim = dataconf['load_dim']
    dim = dataconf['crop_dim']
    opt = dataconf['opt'] if 'opt' in dataconf else {}
    mean = np.asarray(dataconf['mean'], dtype=K.floatx())
    std = np.asarray(dataconf['std'], dtype=K.floatx())

    # load and preprocess data
    print datapath
    (X_train, y_train), (X_val, y_val), num_to_name = load_data(datapath, 
        new_size=load_dim, p_train=0.8, seed=SEED)
    num_classes = len(num_to_name)

    X_train = K.cast_to_floatx(X_train)
    X_train -= mean.reshape(1, 1, 3)
    X_train /= std.reshape(1, 1, 3)

    X_val = K.cast_to_floatx(X_val)
    X_val -= mean.reshape(1, 1, 3)
    X_val /= std.reshape(1, 1, 3)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    train, val = (X_train, y_train), (X_val, y_val)

    # load model
    print "Load model..."
    if args.model == 'resnet':
        model = ResnetBuilder.build_resnet_18((3, dim, dim), num_classes)
    elif args.model == 'vgg16':
        model = keras.applications.vgg16.VGG16(weights=None, classes=num_classes)

    optimizer = keras.optimizers.SGD(**opt)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print "Training samples =", X_train.shape
    print "Mean = {}, std = {}".format(mean, std)
    print "Number of classes =", num_classes
    print "Training parameters =", optimizer.get_config()
    
    Trainer.run(model, train, val, num_classes, dim=dim, num_epochs=args.epochs)