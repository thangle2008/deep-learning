import os
import numpy as np
from functools import partial

import keras.backend as K
from keras.utils import to_categorical

from .helpers import DirectoryDataGenerator
from utils.imgprocessing import meanstd, center_crop, random_crop, horizontal_flip


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
    train_transforms = [
        partial(meanstd, mean=mean, std=std),
        partial(random_crop, new_size=CROP_DIM),
        partial(horizontal_flip, f=0.5),
    ]

    # data generators
    train_generator = DirectoryDataGenerator(
        os.path.join(URL, 'train'), train_transforms, shuffle=True)

    val_generator = get_test_gen('val')

    return (train_generator, val_generator)


def get_test_gen(datatype='val'):

    transforms = [
        partial(meanstd, mean=mean, std=std),
        partial(center_crop, new_size=CROP_DIM)
    ]

    generator = DirectoryDataGenerator(
        os.path.join(URL, datatype), transforms, shuffle=False)

    return generator