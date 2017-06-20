import os, shutil

import scipy.io as sio
import numpy as np
from functools import partial

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from .helpers import get_transform, get_augmented_generator
from utils.imgprocessing import meanstd, center_crop, random_crop
from utils.imgloader import load_data


mean = np.asarray([119.26753706, 115.92306357, 116.10504895], dtype=K.floatx())
std = np.asarray([75.48790007, 75.23135039, 77.03315339], dtype=K.floatx())

LOAD_DIM = 256
CROP_DIM = 224


def split_to_classes(folder, annotation_file):
    """
    Split a folder of images to a subdirectory per class 
    based on a MATLAB annotation file.
    """
    # get annotations
    annotations = sio.loadmat(annotation_file)['annotations'][0]

    for ann in annotations:

        label = ann[4].flatten()[0]
        fname = ann[5].flatten()[0]

        imgpath = os.path.join(folder, fname)
        subfolder = os.path.join(folder, "{0:0>3}".format(label))

        # check if the subdirectory for this folder already exists
        if not os.path.isdir(subfolder):
            os.makedirs(subfolder)

        # move image to its subfolder
        shutil.move(imgpath, subfolder)


def get_data_gen():
    """
    Return train and val generators that give data in batches,
    and data label names.
    """

    # load data
    (X_train, y_train), (X_val, y_val), label_names = load_data(
        './data/images/cars/cars_train', 
        new_size=LOAD_DIM, p_train=0.8, scale=False, seed=28)


    X_train = K.cast_to_floatx(X_train)
    X_val = K.cast_to_floatx(X_val)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)


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

    train_generator = train_datagen.flow(X_train, y_train, 
        shuffle=True, seed=28)
    val_generator = val_datagen.flow(X_val, y_val, shuffle=False)


    # generators with transformations
    train_generator = get_augmented_generator(train_generator, 
        train_transform, new_size=CROP_DIM)
    val_generator = get_augmented_generator(val_generator, 
        val_transform, new_size=CROP_DIM)


    metadata = {
        'label_names': label_names,
        'num_train_samples': X_train.shape[0],
        'num_val_samples': X_val.shape[0],
        'dim': CROP_DIM
    }

    return (train_generator, val_generator, metadata)


def get_test_gen():

    (X_test, y_test), _, label_names = load_data(
        './data/images/cars/cars_test', 
        new_size=LOAD_DIM, p_train=0, scale=False)

    X_test = K.cast_to_floatx(X_test)
    y_test = to_categorical(y_test)

    # define preprocessing pipeline
    test_transform = get_transform(
        partial(meanstd, mean=mean, std=std),
        partial(center_crop, new_size=CROP_DIM)
    )

    # data generators
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow(X_test, y_test, shuffle=False)


    # generators with transformations
    test_generator = get_augmented_generator(test_generator, 
        test_transform, new_size=CROP_DIM)

    return test_generator