import cPickle
import numpy as np

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.datasets import cifar10


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

    X_train = K.cast_to_floatx(X_train)
    X_val = K.cast_to_floatx(X_val)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    X_train -= mean
    X_val -= mean

    X_train /= std
    X_val /= std

    # data generators
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train, 
                                         shuffle=True, seed=28)
    val_generator = val_datagen.flow(X_val, y_val, shuffle=False)

    return train_generator, val_generator


def get_test_gen(dtype='val'):

    if dtype != 'val':
        raise ValueError("CIFAR10 only has validation data.")

    _, (X_val, y_val) = cifar10.load_data()

    X_val = K.cast_to_floatx(X_val)
    y_val = to_categorical(y_val)

    X_val -= mean
    X_val /= std

    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(X_val, y_val, shuffle=False)

    return val_generator
