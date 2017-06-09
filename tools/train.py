from __future__ import division

import random
import json
import numpy as np

import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from utils.imgprocessing import crop, horizontal_flip

from models.resnet import ResnetBuilder
from models import vgg16


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


def run(model, train, val, num_classes, batch_size=32, 
        dim=224, num_epochs=100, opt={}):
    """
    Train a classifier with the provided training and validation data.
    The model must be a compiled keras model.
    """
    # set the random seed for image processing and shuffling
    random.seed(28)
    np.random.seed(28)

    X_train, y_train = train
    X_val, y_val = val
    
    if K.image_data_format() == 'channels_first':
        X_train = X_train.transpose(0, 3, 1, 2)
        X_val = X_val.transpose(0, 3, 1, 2)

    # callbacks list
    model_checkpoint = ModelCheckpoint(
        'weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=5, verbose=1, min_lr=0.0001)

    # define generator functions for training and validation data
    train_transform = _get_transform_func(dim, 'random', True)
    val_transform = _get_transform_func(dim, 'center', False)

    train_gen = _batch_generator(X_train, y_train, 
        batch_size=batch_size, shuffle=True, augment_func=train_transform)
    val_gen = _batch_generator(X_val, y_val, 
        batch_size=batch_size, shuffle=False, augment_func=val_transform)

    # load model
    print "Load model..."
    if model == 'resnet':
        model = ResnetBuilder.build_resnet_18((3, dim, dim), num_classes)
    elif model == 'vgg16':
        model = vgg16.build_model(weights=None, input_shape=(dim, dim, 3), 
            classes=num_classes)

    optimizer = keras.optimizers.SGD(**opt)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print "Number of parameters =", model.count_params()
    print "Training samples =", X_train.shape
    print "Number of classes =", num_classes
    print "Training parameters =", optimizer.__class__, optimizer.get_config()

    model.fit_generator(
        train_gen,
        steps_per_epoch=X_train.shape[0] // batch_size,
        epochs=num_epochs,
        validation_data=val_gen,
        validation_steps=X_val.shape[0] // batch_size,
        callbacks=[model_checkpoint, reduce_lr],
    )

    print "Best validation loss = {:.3f}".format(model_checkpoint.best)
    return model_checkpoint.best
