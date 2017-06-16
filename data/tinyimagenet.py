import numpy as np
from functools import partial

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from .helpers import get_transform, get_augmented_generator
from utils.imgprocessing import meanstd, center_crop, random_crop
from utils.imgloader import load_imagenet


def get_data_gen():
    """
    Return train and val generators that give data in batches,
    and data label names.
    """

    # load data
    (X_train, y_train), (X_val, y_val), label_names = \
        load_imagenet('./data/images/tiny-imagenet-200')


    X_train = K.cast_to_floatx(X_train)
    X_val = K.cast_to_floatx(X_val)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)


    # define preprocessing pipeline
    mean = np.asarray([120.94365593, 113.68772887, 99.6798221], dtype=K.floatx())
    std = np.asarray([68.73304916, 66.44590252, 69.13904857], dtype=K.floatx())


    train_transform = get_transform(
        partial(meanstd, mean=mean, std=std),
        partial(random_crop, new_size=56)
    )

    val_transform = get_transform(
        partial(meanstd, mean=mean, std=std),
        partial(center_crop, new_size=56)
    )


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


    # generators with transformations
    train_generator = get_augmented_generator(train_generator, 
        train_transform, new_size=56)
    val_generator = get_augmented_generator(val_generator, 
        val_transform, new_size=56)


    metadata = {
        'label_names': label_names,
        'num_train_samples': X_train.shape[0],
        'num_val_samples': X_val.shape[0],
        'dim': 56
    }

    return (train_generator, val_generator, metadata)


def get_optimizer():
    """
    Return the params for training with SGD.
    """

    return {
        'lr': 0.1,
        'nesterov': True,
        'momentum': 0.9
    }