from functools import partial
import numpy as np

import keras.backend as K
from keras.utils import to_categorical
from keras.datasets import cifar10

from utils.datagen import ArrayDataGenerator
from utils.imgprocessing import random_crop, horizontal_flip


TRAIN_DIM = 32
NUM_CLASSES = 10

# define preprocessing pipeline
mean = np.asarray([125.30691805, 122.95039414, 113.86538318], dtype=K.floatx())
std = np.asarray([62.99321928, 62.08870764, 66.70489964], dtype=K.floatx())


def get_data_gen():
    """
    Return train and val generators that give data in batches,
    and data label names.
    """

    # load data
    (X_train, y_train), (X_val, y_val) = cifar10.load_data()

    if K.image_data_format() == 'channels_first':
        X_train = X_train.transpose(0, 2, 3, 1)
        X_val = X_val.transpose(0, 2, 3, 1)

    X_train = K.cast_to_floatx(X_train)
    X_val = K.cast_to_floatx(X_val)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    X_train -= mean
    X_val -= mean

    X_train /= std
    X_val /= std

    # train generator
    train_transforms = [
        partial(random_crop, new_size=32, padding=4),
        partial(horizontal_flip, f=0.5)
    ]

    train_generator = ArrayDataGenerator(X_train, y_train,
                                         transforms=train_transforms,
                                         shuffle=True, seed=28)

    # validation generator
    val_generator = ArrayDataGenerator(X_val, y_val, shuffle=False)

    return train_generator, val_generator


def get_test_gen(dtype='val'):

    if dtype != 'val':
        raise ValueError("CIFAR10 only has validation data.")

    _, (X_val, y_val) = cifar10.load_data()

    if K.image_data_format() == 'channels_first':
        X_val = X_val.transpose(0, 2, 3, 1)

    X_val = K.cast_to_floatx(X_val)
    y_val = to_categorical(y_val)

    X_val -= mean
    X_val /= std

    val_generator = ArrayDataGenerator(X_val, y_val, shuffle=False)

    return val_generator
