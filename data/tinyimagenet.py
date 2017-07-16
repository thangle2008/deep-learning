import os
import shutil

import numpy as np
from functools import partial

import keras.backend as K

from utils.datagen import DirectoryDataGenerator
from utils.imgprocessing import (meanstd, color_jitter,
                                 center_crop, random_crop, horizontal_flip)


mean = np.asarray([120.94365593, 113.68772887, 99.6798221], dtype=K.floatx())
std = np.asarray([68.73304916, 66.44590252, 69.13904857], dtype=K.floatx())

LOAD_DIM = 64
CROP_DIM = TRAIN_DIM = 56
NUM_CLASSES = 200

URL = './data/images/tiny-imagenet-200'


def get_data_gen():
    """
    Return train and val generators that give data in batches,
    and data label names.
    """
    # define preprocessing pipeline
    train_transforms = [
        partial(color_jitter, brightness=0.4, contrast=0.4, saturation=0.4),
        partial(meanstd, mean=mean, std=std),
        partial(random_crop, new_size=CROP_DIM, padding=4),
        partial(horizontal_flip, f=0.5),
    ]    

    # data generators
    train_generator = DirectoryDataGenerator(
        os.path.join(URL, 'train'), train_transforms, shuffle=True, seed=28)

    val_generator = get_test_gen('val')

    return train_generator, val_generator


def get_test_gen(datatype):

    if datatype == 'train':
        path = os.path.join(URL, 'train')
    else:
        path = os.path.join(URL, datatype, 'images')

    transforms = [
        partial(center_crop, new_size=CROP_DIM),
        partial(meanstd, mean=mean, std=std),
    ]

    generator = DirectoryDataGenerator(path, transforms, shuffle=False)

    return generator
