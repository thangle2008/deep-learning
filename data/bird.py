import os
import numpy as np
from functools import partial

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from .helpers import get_transform, get_augmented_generator
from utils.imgprocessing import meanstd, center_crop, random_crop


mean = np.asarray([81.2651068 , 156.35841347, 126.5182145], dtype=K.floatx())
std = np.asarray([66.50549267, 55.71497419, 50.10397097], dtype=K.floatx())

LOAD_DIM = 140
CROP_DIM = TRAIN_DIM = 128
URL = './data/images/all_years_140x140'


def get_data_gen():
    """
    Return train and val generators that give data in batches,
    and data label names.
    """
    # define preprocessing pipeline
    train_transform = get_transform(
        partial(meanstd, mean=mean, std=std),
        partial(random_crop, new_size=CROP_DIM)
    )

    val_transform = get_transform(
        partial(meanstd, mean=mean, std=std),
        partial(center_crop, new_size=CROP_DIM)
    )

    # data generators
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        os.path.join(URL, 'train'), target_size=(LOAD_DIM, LOAD_DIM),
        shuffle=True, seed=28)
    val_generator = val_datagen.flow_from_directory(
        os.path.join(URL, 'val'), target_size=(LOAD_DIM, LOAD_DIM),
        shuffle=False)


    metadata = {
        'label_names': train_generator.class_indices,
        'num_train_samples': train_generator.n,
        'num_val_samples': val_generator.n,
        'dim': TRAIN_DIM
    }

    # generators with transformations
    train_generator = get_augmented_generator(train_generator, 
        train_transform, new_size=TRAIN_DIM)
    val_generator = get_augmented_generator(val_generator, 
        val_transform, new_size=TRAIN_DIM)

    return (train_generator, val_generator, metadata)


def get_test_gen():

    val_transform = get_transform(
        partial(meanstd, mean=mean, std=std),
        partial(center_crop, new_size=CROP_DIM)
    )

    val_datagen = ImageDataGenerator()

    val_generator = val_datagen.flow_from_directory(
        os.path.join(URL, 'test'), target_size=(LOAD_DIM, LOAD_DIM),
        shuffle=False)

    num_samples = val_generator.n

    val_generator = get_augmented_generator(val_generator, 
        val_transform, new_size=TRAIN_DIM)

    return (val_generator, num_samples)