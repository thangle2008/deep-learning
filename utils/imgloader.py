import os
import re
import multiprocessing

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.misc import imread, imresize

from .imgprocessing import resize_and_crop


def get_paths_with_labels(folder):
    """
    Return a list of file paths with labels in a directory.

    Args:
        folder: Path to the folder. Note that the folder must have
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

    Returns:
        A tuple of file paths, corresponding labels, and label names (
        label names are just the names of the subdirectories, in the order
        they were loaded in).
    """
    filepaths = []
    categories = []
    
    label_names = [] # class num to class name

    n_class = 0
    # get the file paths
    for root, dirnames, filenames in os.walk(folder):  
        dirnames.sort()
        if root == folder:
            continue

        class_id = root.split('/')[-1]
        
        for filename in filenames:
            if re.search('(?i)\.(jpg|png|jpeg)$', filename):
                filepath = os.path.join(root, filename)
                filepaths.append(filepath)
                categories.append(n_class)

        label_names.append(class_id)
        n_class += 1

    return filepaths, categories, label_names


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


def _load_img(path, n_class, new_size=None, scale=True, mode='RGB'):
    img = imread(path, mode=mode)

    if new_size is not None:
        if scale:
            img = resize_and_crop(img, new_size)
        else:
            img = imresize(img, (new_size, new_size))

    return img, n_class


def _multi_load_img(paths, labels, new_size=None, scale=True):
    """
    Loads images from paths asynchronously. Also preserves the labels.
    """

    pool = multiprocessing.Pool()
    num_samples = len(paths)
    results = [pool.apply_async(_load_img, (paths[i], labels[i], new_size, scale)) 
               for i in xrange(num_samples)]
    imgs = [r.get() for r in results]

    X, y = zip(*imgs)
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def load_data(folder, p_train=0.5, new_size=None, 
              scale=True, seed=None):
    """
    Loads data from a folder and returns the tuple (train, test, label_names).
    Expects the folder to have a subdirectory per class.
    """

    paths, labels, label_names = get_paths_with_labels(folder)

    paths = np.asarray(paths)
    labels = np.asarray(labels)

    # load images
    if p_train <= 0 or p_train >= 1:
        train = _multi_load_img(paths, labels, new_size=new_size)
        test = None
    else:
        train, test = _split_data(paths, labels, p_train, seed)
        train = _multi_load_img(train[0], train[1], new_size=new_size, scale=scale)
        test = _multi_load_img(test[0], test[1], new_size=new_size, scale=scale)

    print train[0].shape    

    return train, test, label_names
