import os, re
import multiprocessing

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.misc import imread

from .imgprocessing import resize_and_crop

def _get_paths(folder):
    """
    Returns list of file paths by classes in a given directory.
    """
    filepaths = []
    categories = []
    
    num_to_name = dict() # class num to class name

    n_class = 0
    # get the file paths
    for root, dirnames, filenames in os.walk(folder):  
        dirnames.sort()
        if root == folder:
            continue
        current_dir = root.split('/')[-1]

        print current_dir
        for filename in filenames:
            if re.search('\.(jpg|png|jpeg)$', filename):
                filepath = os.path.join(root, filename)
                filepaths.append(filepath)
                categories.append(n_class)

        num_to_name[n_class] = current_dir
        n_class += 1

    return filepaths, categories, num_to_name


def _split_data(X, y, p_train=0.5, seed=None):
    """
    Splits data into train and test data.
    X contains the data and y contains the labels.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=p_train,
                                 random_state=seed)

    train_index, test_index = next(iter(sss.split(X, y)))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    return (X_train, y_train), (X_test, y_test)


def _load_img(path, n_class, new_size=None, mode='RGB'):
        img = imread(path, mode=mode)
        if new_size:
            img = resize_and_crop(img, new_size)
        return img, n_class


def load_data(folder, p_train=0.5, new_size=None, seed=None):
    """
    Loads data from a folder and returns the tuple (train, test, num_to_name).
    If seed is None, then splitting is nondeterministic (not the same every run).
    """

    X, y, num_to_name = _get_paths(folder)

    # load images
    pool = multiprocessing.Pool()
    num_samples = len(X)
    results = [pool.apply_async(_load_img, (X[i], y[i], new_size)) 
                    for i in xrange(num_samples)]
    imgs = [r.get() for r in results]

    X, y = zip(*imgs)
    X = np.asarray(X)
    y = np.asarray(y)

    if p_train <= 0 or p_train >= 1:
        return (X, y), None, num_to_name

    # split data
    train, test = _split_data(X, y, p_train, seed)
    return train, test, num_to_name