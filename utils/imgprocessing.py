from __future__ import division
import math

import numpy as np
from scipy.misc import imresize


def center_crop(img, new_size):
    """
    Crop an image at the center. Assume that both dims are not 
    less than new size.
    """
    h, w = img.shape[:2]

    h_offset = int(math.ceil((h - new_size) / 2))
    w_offset = int(math.ceil((w - new_size) / 2))

    return img[h_offset:h_offset+new_size, w_offset:w_offset+new_size]


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

    def __init__(self, data, centered=False, standardized=False):
        self._data = np.asarray(data, dtype=np.float64)
        self._centered = centered
        self._standardized = standardized
        self._completed = False

    def process(self):
        data = self._data

        if self._centered:
            mean = np.mean(data, axis=(0, 1, 2)) # compute means across color channels
            self._mean = mean.reshape((1, 1, 3))
            data -= self._mean 

        if self._standardized:
            std = np.std(data, axis=(0, 1, 2))
            self._std = std.reshape((1, 1, 3))
            data /= self._std

        self._completed = True

    def get_info(self):
        res = dict()
        if self._completed:
            if self._centered: 
                res['mean'] = self._mean
            if self._standardized:
                res['std'] = self._std
        return res