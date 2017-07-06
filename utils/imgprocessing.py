from __future__ import division
import math
import random

import numpy as np
from scipy.misc import imresize
import skimage.transform

from keras.preprocessing.image import ImageDataGenerator


def meanstd(img, mean=None, std=None):
    """Centralize and normalize an image."""

    if mean is not None:
        img -= mean.reshape(1, 1, 3)

    if std is not None:
        img /= std.reshape(1, 1, 3) 

    return img


def crop(img, new_size, method='center'):
    """Crop an image to a new size."""
    if method == 'center':
        return center_crop(img, new_size)
    elif method == 'random':
        return random_crop(img, new_size)
    else:
        raise ValueError


def center_crop(img, new_size):
    """
    Crop an image at the center. Assume that both dims are not 
    less than new size.
    """
    h, w = img.shape[:2] 

    h_offset = int(math.ceil((h - new_size) / 2))
    w_offset = int(math.ceil((w - new_size) / 2))

    return img[h_offset:h_offset+new_size, w_offset:w_offset+new_size]


def ten_crop(img, new_size):

    img_list = [
        img[0:new_size, 0:new_size],  # top left corner
        img[0:new_size, -new_size:],  # top right corner
        img[-new_size:, 0:new_size],  # bottom left corner
        img[-new_size:, -new_size:],  # bottom right corner
        center_crop(img, new_size)    # center
    ]

    res = []

    # get the horizontal flip
    for img in img_list:
        res.append(img)
        res.append(img[:, ::-1])

    return np.asarray(res)


def random_crop(img, new_size):
    """
    Randomly choose a region from an image to crop from.
    """
    h, w = img.shape[:2]

    h_offset = random.randint(0, h-new_size)
    w_offset = random.randint(0, w-new_size)

    return img[h_offset:h_offset+new_size, w_offset:w_offset+new_size]


def scale(img, new_size):
    """
    Scale an image to the given size.
    """
    img = skimage.transform.resize(img, (new_size, new_size), 
                                   preserve_range=True)
    
    return img


def horizontal_flip(img, f=0.5):
    """
    Randomly flip an image horizontally.
    """
    num = random.random()
    if num < f:
        img = img[:, ::-1]
    return img


def resize_and_crop(img, new_size, interp='bicubic'):
    """
    Firstly, resize the smaller dimension of the image to the new size.
    Then, crop the image at the center.
    """
    h, w = img.shape[:2]
    aspect_ratio = h / w

    # resize the smaller dim to new size and keep aspect ratio
    if h < w:
        new_img = imresize(img, (new_size, int(new_size / aspect_ratio)), interp)
    else:
        new_img = imresize(img, (int(new_size * aspect_ratio), new_size), interp)

    # crop the image to the new size
    return center_crop(new_img, new_size)


def width_shift(img, shift_range):
    """
    Shift the image horizontally.
    """

    datagen = ImageDataGenerator(width_shift_range=shift_range)
    img_batch = img.reshape((1,) + img.shape)
    return next(datagen.flow(img_batch, batch_size=1, shuffle=False))[0]


def height_shift(img, shift_range):
    """
    Shift the image vertically.
    """

    datagen = ImageDataGenerator(height_shift_range=shift_range)
    img_batch = img.reshape((1,) + img.shape)
    return next(datagen.flow(img_batch, batch_size=1, shuffle=False))[0]