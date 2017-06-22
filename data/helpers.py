import numpy as np
import keras.backend as K


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
