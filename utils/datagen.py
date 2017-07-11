import numpy as np

import keras.backend as K
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from utils.imgloader import get_paths_with_labels

from scipy.misc import imread


def get_transform(*functions):
    """
    Return a function composed from a list of functions.
    """

    def transform(img):
        for f in functions:
            img = f(img)

        return img

    return transform


def get_augmented_generator(gen, transform_func, new_size=None):
    """
    Yield batches from a given generator with specified transformations.
    """
    for X_batch, y_batch in gen:

        if new_size is None:
            X_batch_new = np.zeros(X_batch.shape, dtype=K.floatx())
        else:
            X_batch_new = np.zeros(
                (X_batch.shape[0], new_size, new_size, 3), dtype=K.floatx())

        for idx in xrange(len(X_batch)):
            X_batch_new[idx] = transform_func(X_batch[idx])

        yield X_batch_new, y_batch


class ArrayDataGenerator(object):
    """Generator that transforms and yields data batches from Numpy arrays.

    This class is a wrapper around the ImageDataGenerator class of Keras
    image preprocessing module.

    Args:
        X: Numpy array of data samples.
        y: Numpy array of data labels.
        transforms: A list of transformation functions. Each of them takes
            an image as input and returns a transformed image.
        shuffle: If true, shuffle the data samples after all of them have
            been loaded before starting the next batch.
        batch_size: The size of each batch.
        seed: This is used to set the seed before shuffling data.

    """

    def __init__(self, X, y, transforms=None, shuffle=True, batch_size=32,
                 seed=None):

        if transforms is None:
            transforms = []

        self.n = X.shape[0]
        self.batch_size = batch_size

        datagen = ImageDataGenerator(data_format='channels_last')
        self._data_generator = datagen.flow(X, y, batch_size=batch_size,
                                            shuffle=shuffle, seed=seed)
        self._transform_func = get_transform(*transforms)
        self.output_shape = self._transform_func(X[0]).shape

        if K.image_data_format() == 'channels_first':
            self.output_shape = (self.output_shape[2],
                                 self.output_shape[0],
                                 self.output_shape[1])

    def __iter__(self):
        return self

    def next(self):
        X, y = next(self._data_generator)
        X_new = np.zeros((X.shape[0], ) + self.output_shape, dtype=K.floatx())

        for idx in xrange(len(X)):
            transformed_img = self._transform_func(X[idx])
            if K.image_data_format() == 'channels_first':
                transformed_img = transformed_img.transpose(2, 0, 1)
            X_new[idx] = transformed_img

        return X_new, y


class DirectoryDataGenerator(object):
    """Generator that transforms and yields data batches from a directory.

    The generator will first load the images, then put them through a list
    of functions (such as cropping, flipping, ...) to produce augmented data.

    Args:
        folder (str): Path to the folder. Note that the folder must have
            a subdirectory per class. For example, one valid directory
            structure is:
                folder/
                    class1/
                        img1.jpg
                        img2.jpg
                        ...
                    class2/
                        img1.jpg
                        img2.jpg
                        ...
        transforms (list): A list of functions, each takes an image
            (as a numpy array) and returns a transformed image.
        shuffle: If true, shuffle the data samples after all of them have
            been loaded before starting the next batch.
        batch_size: The size of each batch.
        seed: This is used to set the seed before shuffling data.

    Attributes:
        n (int): the total number of images in the directory.
        batch_size (int): the size of each batch (note that the last batch
            can have different size than those of the others since the total
            number of images may not be divisible by the batch size).
        paths (list): The paths of all the images.

    """

    def __init__(self, folder, transforms, shuffle=True, batch_size=32,
                 seed=None):

        paths, labels, label_names = get_paths_with_labels(folder)

        self.n = len(paths)
        self.paths = np.asarray(paths)
        self.labels = to_categorical(labels, num_classes=len(label_names))
        self.label_names = label_names

        self.shuffle = shuffle
        self.seed = seed

        self.transform = get_transform(*transforms)

        self.batch_size = batch_size
        self.batch_idx = 0
        self.num_batches_so_far = -1
        self.indices = np.arange(self.n)

        # calculate output shape by loading an image and
        # passing it through the functions
        img = imread(paths[0])
        img = np.asarray(img, dtype=K.floatx())
        self.output_shape = self.transform(img).shape

        if K.image_data_format() == 'channels_first':
            self.output_shape = (self.output_shape[2],
                                 self.output_shape[0],
                                 self.output_shape[1])

        self.reset()

    def reset(self):
        """
        Reset the generator to start another batch.
        """

        self.batch_idx = 0
        self.num_batches_so_far += 1

        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed + self.num_batches_so_far)
            self.indices = np.random.permutation(self.n)

    def next(self):
        """
        Get the next batch.
        """

        if self.batch_idx >= self.n:
            self.reset()

        # incomplete batch
        if self.batch_idx + self.batch_size >= self.n:
            batch_size = self.n - self.batch_idx
        else:
            batch_size = self.batch_size

        indices = self.indices[self.batch_idx:self.batch_idx + batch_size]
        paths, labels = self.paths[indices], self.labels[indices]

        x_batch = np.zeros((batch_size,) + self.output_shape, dtype=K.floatx())

        # load and process the image batch
        for idx in xrange(len(paths)):
            img = imread(paths[idx], mode='RGB')
            img = np.asarray(img, dtype=K.floatx())
            img = self.transform(img)
            if K.image_data_format() == 'channels_first':
                img = img.transpose(2, 0, 1)
            x_batch[idx] = img

        self.batch_idx += self.batch_size

        return x_batch, labels

    def __iter__(self):
        return self
