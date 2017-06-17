import os, re
import multiprocessing

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.misc import imread

from .imgprocessing import resize_and_crop

### General format image loading functions

def _get_paths_with_labels(folder, dformat=None):
    """
    Returns list of file paths by classes in a given directory.
    The directory is expected to have a subdirectory per class.
    """
    filepaths = []
    categories = []
    
    label_names = [] # class num to class name

    class_name = None
    n_class = 0
    # get the file paths
    for root, dirnames, filenames in os.walk(folder):  
        dirnames.sort()
        if root == folder:
            continue

        current_dir = root.split('/')[-1]
        if not dformat:
            class_id = current_dir
        elif dformat == 'imagenet' and current_dir != 'images':
            class_id = current_dir
            continue

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


def _load_img(path, n_class, new_size=None, mode='RGB'):
    img = imread(path, mode=mode)
    if new_size:
        img = resize_and_crop(img, new_size)
    return img, n_class


def _multi_load_img(paths, labels, new_size=None):
    """
    Loads images from paths asynchronously. Also preserves the labels.
    """

    pool = multiprocessing.Pool()
    num_samples = len(paths)
    results = [pool.apply_async(_load_img, (paths[i], labels[i], new_size)) 
                    for i in xrange(num_samples)]
    imgs = [r.get() for r in results]

    X, y = zip(*imgs)
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def load_data(folder, dformat=None, p_train=0.5, new_size=None, seed=None):
    """
    Loads data from a folder and returns the tuple (train, test, label_names).
    If seed is None, then splitting is nondeterministic (not the same 
    every run).
    """

    paths, labels, label_names = _get_paths_with_labels(folder, 
                                                        dformat=dformat)

    paths = np.asarray(paths)
    labels = np.asarray(labels)

    # load images
    if p_train <= 0 or p_train >= 1:
        train = _multi_load_img(paths, labels, new_size=new_size)
        test = None
    else:
        train, test = _split_data(paths, labels, p_train, seed)
        train = _multi_load_img(train[0], train[1], new_size=new_size)
        test = _multi_load_img(test[0], test[1], new_size=new_size)

    print train[0].shape    

    return train, test, label_names


### ImageNet format loading functions

def _read_annotations(filepath):
    """
    Read an annotation file and return a dictionary containing
    each image's information.
    """
    annotations = dict()
    with open(filepath, 'r') as annotation_file:
        for line in annotation_file:
            fname, class_id, bx, by, tx, ty = line.split()
            bbox = (bx, by, tx, ty) 
            annotations[fname] = {'id': class_id, 'bbox': bbox}
    return annotations


def _get_val_imagenet_paths(folder, id_to_num):
    """
    Get validation image paths.
    id_to_name is a dictionary that maps a class id to a label number.
    """
    # get annotation file
    annotation_file = os.path.join(folder, 'val_annotations.txt')
    annotations = _read_annotations(annotation_file)

    # get image paths with labels
    paths, labels = [], []

    filedir = os.path.join(folder, 'images')

    for f in os.listdir(filedir):
        p = os.path.join(filedir, f)
        if os.path.isfile(p) and re.search('(?i)\.(jpg|png|jpeg)', f):
            paths.append(p)
            labels.append( id_to_num[annotations[f]['id']] )

    return paths, labels


def load_imagenet(folder):
    """
    Load images from a folder in ImageNet format.
    """
    # load training data
    train_folder = os.path.join(folder, 'train')
    train, _, label_names = load_data(train_folder, 
                                    dformat='imagenet', 
                                    p_train=1.0)

    # load validation data
    id_to_num = dict((label_names[i], i) for i in range(len(label_names)))
    val_folder = os.path.join(folder, 'val')
    val_paths, val_labels = _get_val_imagenet_paths(val_folder, id_to_num)
    val = _multi_load_img(val_paths, val_labels)

    return train, val, label_names