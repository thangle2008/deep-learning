import numpy as np
from functools import partial

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from .helpers import get_transform, get_augmented_generator
from utils.imgprocessing import meanstd, center_crop, random_crop
from utils.imgloader import load_data


def get_data_gen():
    """
    Return train and val generators that give data in batches,
    and data label names.
    """

    # load data
    (X_train, y_train), (X_val, y_val), label_names = load_data(
        './data/images/all_years_342x256', new_size=256, p_train=0.8, seed=28)


    X_train = K.cast_to_floatx(X_train)
    X_val = K.cast_to_floatx(X_val)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)


    # define preprocessing pipeline
    mean = np.asarray([66.50691593, 159.22152607, 125.28014714], dtype=K.floatx())
    std = np.asarray([67.49946853, 57.54205911, 52.35912736], dtype=K.floatx())


    train_transform = get_transform(
        partial(meanstd, mean=mean, std=std),
        partial(random_crop, new_size=224)
    )

    val_transform = get_transform(
        partial(meanstd, mean=mean, std=std),
        partial(center_crop, new_size=224)
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
        train_transform, new_size=224)
    val_generator = get_augmented_generator(val_generator, 
        val_transform, new_size=224)


    metadata = {
        'label_names': label_names,
        'num_train_samples': X_train.shape[0],
        'num_val_samples': X_val.shape[0],
        'dim': 224
    }

    return (train_generator, val_generator, metadata)
