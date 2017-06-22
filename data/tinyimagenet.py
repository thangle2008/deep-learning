import os, shutil

import numpy as np
from functools import partial

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from .helpers import get_transform, get_augmented_generator
from utils.imgprocessing import meanstd, center_crop, random_crop, scale


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
    train_transform = get_transform(
        partial(meanstd, mean=mean, std=std),
        partial(random_crop, new_size=CROP_DIM),
        partial(scale, new_size=TRAIN_DIM)
    )

    val_transform = get_transform(
        partial(meanstd, mean=mean, std=std)
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
        os.path.join(URL, 'val', 'images'), target_size=(LOAD_DIM, LOAD_DIM),
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
