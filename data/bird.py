import os
import numpy as np
from functools import partial

import keras.backend as K

from utils.datagen import DirectoryDataGenerator
from utils.imgprocessing import meanstd, center_crop, random_crop, \
    horizontal_flip, ten_crop, color_jitter, resize_and_crop


mean = np.asarray([69.47252731, 160.22688271, 126.6051936], dtype=K.floatx())
std = np.asarray([71.34189641, 57.88649865, 53.74540484], dtype=K.floatx())

LOAD_DIM = 256
CROP_DIM = TRAIN_DIM = 224
NUM_CLASSES = 14
URL = './data/images/all_years_342x256'


def get_data_gen():
    """
    Return train and val generators that give data in batches,
    and data label names.
    """
    # define preprocessing pipeline
    train_transforms = [
        partial(resize_and_crop, new_size=LOAD_DIM),
        partial(color_jitter, brightness=0.4, contrast=0.4, saturation=0.4),
        partial(meanstd, mean=mean, std=std),
        partial(random_crop, new_size=CROP_DIM),
        partial(horizontal_flip, f=0.5),
    ]

    # data generators
    train_generator = DirectoryDataGenerator(
        os.path.join(URL, 'train'), train_transforms, shuffle=True)
    
    val_generator = get_test_gen('val')

    return train_generator, val_generator


def get_test_gen(datatype='val'):

    crop = ten_crop if datatype == 'test' else center_crop

    transforms = [
        partial(resize_and_crop, new_size=LOAD_DIM),
        partial(meanstd, mean=mean, std=std),
        partial(crop, new_size=CROP_DIM)
    ]

    generator = DirectoryDataGenerator(
        os.path.join(URL, datatype), transforms, shuffle=False)

    return generator
