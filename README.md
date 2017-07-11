# DeepDetection

## Overview

This repository contains a framework for training Resnet with real-time data augmentation in Keras. 

The Resnet source code was taken from <https://github.com/raghakot/keras-resnet> with some modifications to allow easy configuration. I also added 
shortcut option A (described in the original paper <https://arxiv.org/pdf/1512.03385.pdf>) for handling residual blocks that
have input and output with different dimensions.

## Dependencies

* Keras (2.0.4)
* Tensorflow (recommended) or Theano
* Numpy, Scipy, Scikit-learn

## Demo

To start training on the CIFAR10 dataset with Resnet-18, simply run:

`$ python main.py cifar10 --depth 20 --filters 16 --shortcut A`

## Usage instructions

### 1) Define a data generator file:

As you have noticed, the `main.py` script takes a required positional argument (cifar10 in the Demo) which specifies the module
to load data from. The `cifar10` module is defined in the `/data` folder. If you want to define your own data generator file, 
here are some guidelines:

1. Firstly, the file must contain a function named `get_data_gen`. This function has no parameters and return 2 data generators,
one for training and one for validation. There are 2 convenient classes for creating these generators in `/utils/datagen.py`:
    1. `ArrayDataGenerator`: takes 2 Numpy arrays (samples and labels) and return a generator that yield data batches 
    from those arrays. Note that the labels must be one-hot (you can convert from a normal label array to one-hot array
    by using keras.utils.to_categorical).
    2. `DirectoryDataGenerator`: similar to ArrayDataGenerator but instead takes the path to the data folder as input. The
    structure of the folder should be:
    ```
    /folder
      /class1
        img1.png
        img2.png
      /class2
        img1.png
        img2.png
      ...
     ```
    The good thing about these classes is that they also have an optional `transforms` parameter, which is a list
    of preprocessing functions to apply on the data images during training and testing. These functions takes an image
    as input and returns the transformed image (If your function needs more parameters, just use the `partial` function from 
    module `functools` or define the processing function as a higher order function). In my code, I use the former way.
    
    Moreover, if you want to create your own generator classes, take a look at `ArrayDataGenerator` and `DirectoryDataGenerator`
    for some ideas. 

2. Secondly, the file must have 2 global variables: TRAIN_DIM (which specifies the dimension of input image to the network)
and NUM_CLASSES (the number of data classes). 

3. (Optional) Similar to `get_data_gen`, you can create a function `get_test_gen` that takes a single parameter, which
specifies whether you want to test the data on validation or test set, and returns a test generator. 

I highly recommend you to inspect the files `/data/cifar10.py` and `/data/tinyimagenet.py` as some examples (you don't need to
care about any other functions in those files that are not described above).

### 2) Training:

Run the main.py script as follows to train on Resnet (you can inspect the main.py script to find out about the usage of every optional argument):

`$ python main.py [data generator file] --model resnet --depth [18, 20, 34, ...]`

During training, the program will save the best model (with lowest validation loss) to the file `best_model.hdf5` and log 
the training steps to `training.csv`. 

To resume training on the saved model, run:

`$ python main.py [data generator file] --resume [path to model hdf5 file] --initial_epoch [the first epoch to be indexed from]`

To plot the training log file, run:

`$ python plot.py [path to csv file]`

### 3) Testing

This assumes that you have defined the function `get_test_gen` in the data generator file. 

`$ python main.py [data generator file] --evaluate ['val', 'test'] --classifier [path to model hdf5 file]`

If you use the method `ten_crop` defined in `/utils/imgprocessing.py` in the list of transformation functions (see `bird.py`)
, you must include the `--ten_crop` flag.

