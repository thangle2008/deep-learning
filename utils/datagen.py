import numpy as np

import keras.backend as K
from keras.utils import to_categorical

from utils.imgloader import get_paths_with_labels

from scipy.misc import imread


def get_transform(*functions):
    """
    Return a transformation function composed from
    a list of transformation functions
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


class DirectoryDataGenerator(object):
    """
    Generator with transformation.
    """

    def __init__(self, folder, transforms, shuffle=True, batch_size=32,
                 seed=28):

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

        self.reset()

    def reset(self):
        """
        Reset the generator to start another batch.
        """

        self.batch_idx = 0
        self.num_batches_so_far += 1

        if self.shuffle:
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

        for idx in xrange(len(paths)):
            img = imread(paths[idx], mode='RGB')
            img = np.asarray(img, dtype=K.floatx())
            img = self.transform(img)
            x_batch[idx] = img

        self.batch_idx += self.batch_size

        return x_batch, labels

    def __iter__(self):
        return self
