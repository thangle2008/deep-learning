from __future__ import division
import math
import random

import numpy as np
from scipy.misc import imresize


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


def random_crop(img, new_size):
    """
    Randomly choose a region from an image to crop from.
    """
    h, w = img.shape[:2]

    h_offset = random.randint(0, h-new_size)
    w_offset = random.randint(0, w-new_size)

    return img[h_offset:h_offset+new_size, w_offset:w_offset+new_size]


def horizontal_flip(img, f=0.5):
    """
    Randomly flip an image horizontally.
    """
    num = random.random()
    if num >= f:
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
    new_img = None
    if h < w:
        new_img = imresize(img, (new_size, int(new_size / aspect_ratio)), interp)
    else:
        new_img = imresize(img, (int(new_size * aspect_ratio), new_size), interp)

    # crop the image to the new size
    return center_crop(new_img, new_size)


class ImgDataPreprocessing:

    def __init__(self, centered=False, standardized=False):
        self._centered = centered
        self._standardized = standardized
        self.mean = None
        self.std = None


    def process(self, data):
        data = data.astype(np.float64)

        if self._centered:
            self.mean = np.mean(data, axis=(0, 1, 2)) # compute means across color channels
            mean = self.mean.reshape((1, 1, 3))
            data -= mean

        if self._standardized:
            self.std = np.std(data, axis=(0, 1, 2))
            std = self.std.reshape((1, 1, 3))
            data /= std
