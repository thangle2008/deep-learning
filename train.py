from __future__ import division
import time

import numpy as np

import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

from models.resnet import ResnetBuilder
from utils.imgloader import load_data
from utils.imgprocessing import ImgDataPreprocessing, crop, horizontal_flip


def augment_batch(data, new_size=None, crop_method='center', h_flip=False):
    """Augment a batch of images."""

    size = new_size if new_size else data.shape[1]
    new_data = np.zeros((data.shape[0], size, size, 3), dtype=K.floatx())

    for idx in xrange(len(data)):
        img = crop(data[idx], size, method=crop_method)
        img = horizontal_flip(img, f=0.5) if h_flip else img
        new_data[idx] = img

    return new_data


def run(train, val, num_classes, dim=224, num_epochs=100):
    """
    Train a classifier with the provided training and validation data.
    The images should be raw.
    """

    X_train, y_train = train
    X_val, y_val = val

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    # calculate mean and std for normalizing
    idp = ImgDataPreprocessing(centered=True, standardized=True)
    idp.process(X_train)
    mean, std = K.cast_to_floatx(idp.mean), K.cast_to_floatx(idp.std)

    print mean, std

    # preprocess images
    X_train, X_val = K.cast_to_floatx(X_train), K.cast_to_floatx(X_val)
    X_train -= mean.reshape(1, 1, 3)
    X_train /= std.reshape(1, 1, 3)
    X_val -= mean.reshape(1, 1, 3)
    X_val /= std.reshape(1, 1, 3)

    # load model
    model = ResnetBuilder.build_resnet_18((3, dim, dim), num_classes)
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator()

    # training and validating
    for e in range(num_epochs):
        print "Training on epoch {}".format(e+1)

        start_time = time.time()

        total_train_err, total_train_acc = 0.0, 0.0
        num_train_batches = 0

        for X_batch, y_batch in datagen.flow(X_train, y_train, 
                                             batch_size=32, shuffle=True):
            X_batch = augment_batch(X_batch, dim, 'random', True)
            train_err, train_acc = model.train_on_batch(X_batch, y_batch)
            total_train_err += train_err
            total_train_acc += train_acc
            num_train_batches += 1

        print "Training time is {:.3f}s".format(time.time()-start_time)
        print "training err = {:.3f}, training acc = {:.2f}%".format(
                                        total_train_err / num_train_batches,
                                        total_train_acc * 100 / num_train_batches)


if __name__ == '__main__':
    train, val, num_to_class = load_data('data/101_ObjectCategories', new_size=140)
    run(train, val, len(num_to_class), dim=128)