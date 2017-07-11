from __future__ import division
import math
import random
from functools import partial

import numpy as np
from scipy.misc import imresize
import skimage.transform


def meanstd(img, mean=None, std=None):
    """
    Subtract then divide the image by the given mean and
    standard deviation.
    """

    if mean is not None:
        img -= mean.reshape(1, 1, 3)

    if std is not None:
        img /= std.reshape(1, 1, 3) 

    return img


def center_crop(img, new_size):
    """
    Crop an image at the center.
    """
    h, w = img.shape[:2] 

    h_offset = int(math.ceil((h - new_size) / 2))
    w_offset = int(math.ceil((w - new_size) / 2))

    return img[h_offset:h_offset+new_size, w_offset:w_offset+new_size]


def ten_crop(img, new_size):
    """
    Crop an image at 4 corners and at the center. Then, return those
    new 5 images along with their corresponding horizontal flips.
    """

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


def random_crop(img, new_size, padding=0):
    """
    Randomly choose a region from an image to crop from.
    Optionally pad the image with zeros before cropping.
    """
    if padding > 0:
        new_img = np.zeros((img.shape[0] + 2 * padding,
                            img.shape[1] + 2 * padding,
                            img.shape[2]), dtype=img.dtype)
        new_img[padding:-padding, padding:-padding] = img
        img = new_img

    h, w = img.shape[:2]

    h_offset = random.randint(0, h-new_size)
    w_offset = random.randint(0, w-new_size)

    return img[h_offset:h_offset+new_size, w_offset:w_offset+new_size]


def scale(img, new_size, interp='bicubic'):
    """
    Scale the smaller size of an image to a given size and keep aspect ratio.
    """
    h, w = img.shape[:2]
    aspect_ratio = h / w

    # resize the smaller dim to new size and keep aspect ratio
    if h < w:
        new_img = imresize(img, (new_size, int(new_size / aspect_ratio)),
                           interp)
    else:
        new_img = imresize(img, (int(new_size * aspect_ratio), new_size),
                           interp)

    return new_img.astype(img.dtype)


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
    new_img = scale(img, new_size, interp=interp)

    # crop the image to the new size
    return center_crop(new_img, new_size)


# Color jittering functions, which are based on the codes from
# https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua
def _rgb2gray(img):
    """
    Return the grayscale representation of a RGB image.
    """
    new_img = np.zeros(img.shape, dtype=img.dtype)
    gs = np.dot(img, [0.299, 0.587, 0.114])
    new_img[..., 0] = gs
    new_img[..., 1] = np.copy(gs)
    new_img[..., 2] = np.copy(gs)
    return new_img


def _blend(img1, img2, alpha):
    """
    Blend 2 images with an alpha factor.
    """
    return alpha * img1 + (1 - alpha) * img2


def adjust_brightness(img, var):
    """
    Adjust image brightness
    """
    gs = np.zeros(img.shape, dtype=img.dtype)
    alpha = 1.0 + np.random.uniform(-var, var)
    return _blend(img, gs, alpha)


def adjust_contrast(img, var):
    """
    Adjust image contrast.
    """
    gs = _rgb2gray(img)
    gs.fill(gs[0].mean())
    alpha = 1.0 + np.random.uniform(-var, var)
    return _blend(img, gs, alpha)


def adjust_saturation(img, var):
    """
    Adjust image saturation.
    """
    gs = _rgb2gray(img)
    alpha = 1.0 + np.random.uniform(-var, var)
    return _blend(img, gs, alpha)


def color_jitter(img, brightness=0.0, contrast=0.0, saturation=0.0):
    """
    Adjust image brightness, contrast, and saturation in random order.
    """
    transforms = [
        partial(adjust_brightness, var=brightness),
        partial(adjust_contrast, var=contrast),
        partial(adjust_saturation, var=saturation)
    ]

    random.shuffle(transforms)

    new_img = np.copy(img)
    for f in transforms:
        new_img = f(new_img)

    return new_img
