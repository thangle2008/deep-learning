import os, shutil

import numpy as np
from functools import partial

import keras.backend as K
from keras.utils import to_categorical

from .helpers import DirectoryDataGenerator
from utils.imgprocessing import (meanstd, center_crop, random_crop, 
                                 horizontal_flip, scale)


mean = np.asarray([120.94365593, 113.68772887, 99.6798221], dtype=K.floatx())
std = np.asarray([68.73304916, 66.44590252, 69.13904857], dtype=K.floatx())

LOAD_DIM = 64
CROP_DIM = 55
TRAIN_DIM = 64

URL = './data/images/tiny-imagenet-200'


def format_training_folder(folder):
    
    for root, dirnames, filenames in os.walk(folder):

        current_dir = root.split('/')[-1]

        if current_dir != 'images': 
            continue

        for filename in filenames:
            src = os.path.join(root, filename)
            dst = os.path.join(root, '..', filename)
            shutil.move(src, dst)

        os.rmdir(root)


def distribute_images(folder, annotation_file):
    """
    Split a folder of images to a subdirectory per class 
    based on an annotation file.
    """
    with open(annotation_file, 'rb') as f:
        for line in f:
            fname, label, lx, ly, tx, ty = line.split()

            fpath = os.path.join(folder, fname)
            subfolder = os.path.join(folder, label)

            # check if the subdirectory for this folder already exists
            if not os.path.isdir(subfolder):
                os.makedirs(subfolder)

            # move image to its subfolder
            shutil.move(fpath, subfolder)


def get_data_gen():
    """
    Return train and val generators that give data in batches,
    and data label names.
    """
    # define preprocessing pipeline
    train_transforms = [
        partial(meanstd, mean=mean, std=std),
        partial(random_crop, new_size=CROP_DIM),
        partial(scale, new_size=TRAIN_DIM),
        partial(horizontal_flip, f=0.5),
    ]
        
    val_transforms = [
        partial(meanstd, mean=mean, std=std)
    ]

    # data generators
    train_generator = DirectoryDataGenerator(
        os.path.join(URL, 'train'), train_transforms, shuffle=True)

    val_generator = DirectoryDataGenerator(
        os.path.join(URL, 'val', 'images'), val_transforms, shuffle=False)

    return (train_generator, val_generator)
