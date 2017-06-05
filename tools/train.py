from __future__ import division

import json
import numpy as np

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

from utils.imgprocessing import crop, horizontal_flip

from .keras_callbacks import BestModelCheck


def _get_transform_func(new_size=None, crop_method='center', h_flip=False):
    """Return a specific transformation function."""

    def _transform(data):
        size = new_size if new_size else data.shape[1]
        new_data = np.zeros((data.shape[0], size, size, 3), dtype=K.floatx())

        if K.image_data_format() == 'channels_first':
            new_data = new_data.transpose(0, 3, 1, 2)

        for idx in xrange(len(data)):
            img = crop(data[idx], size, method=crop_method, 
                                        img_format=K.image_data_format())
            if h_flip:
                img = horizontal_flip(img, f=0.5, 
                        img_format = K.image_data_format()) 

            new_data[idx] = img

        return new_data

    return _transform


def _batch_generator(X, y, batch_size=32, shuffle=True, augment_func=None):
    """Generate batches from given data."""

    indices = np.arange(X.shape[0])
    n_batches = X.shape[0] // batch_size

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for batch in range(n_batches):
            idxs = indices[batch*batch_size : (batch+1)*batch_size]
            X_batch, y_batch = X[idxs], y[idxs]
            X_batch = augment_func(X_batch)

            yield X_batch, y_batch


def run(model, train, val, num_classes, batch_size=32, dim=224, num_epochs=100):
    """
    Train a classifier with the provided training and validation data.
    The model must be a compiled keras model.
    """

    X_train, y_train = train
    X_val, y_val = val
    
    if K.image_data_format() == 'channels_first':
        X_train = X_train.transpose(0, 3, 1, 2)
        X_val = X_val.transpose(0, 3, 1, 2)

    best_model_check = BestModelCheck('./experiments/weights.hdf5')

    # define generator functions for training and validation data
    train_transform = _get_transform_func(dim, 'random', True)
    val_transform = _get_transform_func(dim, 'center', False)

    train_gen = _batch_generator(X_train, y_train, 
        batch_size=batch_size, shuffle=True, augment_func=train_transform)
    val_gen = _batch_generator(X_val, y_val, 
        batch_size=batch_size, shuffle=False, augment_func=val_transform)

    model.fit_generator(
        train_gen,
        steps_per_epoch=X_train.shape[0] // batch_size,
        epochs=num_epochs,
        validation_data=val_gen,
        validation_steps=X_val.shape[0] // batch_size,
        callbacks=[best_model_check,],
    )

    print "Best validation loss = {:.3f}".format(best_model_check.best_val_loss)
    print "Best validation acc = {:.3f}%".format(
        best_model_check.best_val_acc * 100)

    return best_model_check.best_val_loss
