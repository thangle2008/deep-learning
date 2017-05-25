from __future__ import division
import time
import progressbar

import numpy as np

import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

from models.resnet import ResnetBuilder
from utils.imgloader import load_data
from utils.imgprocessing import ImgDataPreprocessing, crop, horizontal_flip


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
    n_batches = X.shape[0] // 32

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for batch in range(n_batches):
            idxs = indices[batch*batch_size : (batch+1)*batch_size]
            X_batch, y_batch = X[idxs], y[idxs]
            X_batch = augment_func(X_batch)

            yield X_batch, y_batch


def run(train, val, num_classes, dim=224, num_epochs=100):
    """
    Train a classifier with the provided training and validation data.
    The images should be raw.
    """

    # mean and std were calculated using 2000 samples from training set
    mean = np.asarray([137.14349233, 131.30804223, 123.85041327],
                dtype=K.floatx())
    std = np.asarray([78.71011359,  76.94469003,  80.1550091],
                dtype=K.floatx())

    # preprocess images
    X_train, y_train = train
    X_val, y_val = val
    X_train, X_val = K.cast_to_floatx(X_train), K.cast_to_floatx(X_val)

    X_train -= mean.reshape(1, 1, 3)
    X_train /= std.reshape(1, 1, 3)

    X_val -= mean.reshape(1, 1, 3)
    X_val /= std.reshape(1, 1, 3)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    
    if K.image_data_format() == 'channels_first':
        X_train = X_train.transpose(0, 3, 1, 2)
        X_val = X_val.transpose(0, 3, 1, 2)

    # X_val = _augment_batch(X_val, dim, 'center', False)
    print X_train.shape

    # load model
    print "Load model..."
    model = ResnetBuilder.build_resnet_18((3, dim, dim), num_classes)
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print "Number of classes =", num_classes

    train_transform = _get_transform_func(dim, 'random', True)
    val_transform = _get_transform_func(dim, 'center', False)

    model.fit_generator(
        _batch_generator(X_train, y_train, augment_func=train_transform),
        steps_per_epoch=X_train.shape[0] // 32,
        epochs=num_epochs,
        validation_data=_batch_generator(X_val, y_val, augment_func=val_transform),
        validation_steps=X_val.shape[0] // 32,
    )

if __name__ == '__main__':
    train, val, num_to_name = load_data('data/101_ObjectCategories', 
                                        p_train=0.8, new_size=140)
    run(train, val, len(num_to_name), dim=128) 