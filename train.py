from __future__ import division
import time

from keras import optimizers
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import sklearn
import numpy as np

from models.resnet import ResnetBuilder
from utils.imgloader import load_data
from utils.imgprocessing import ImgDataPreprocessing

def data_generator(X, y, batch_size, shuffle=False):
    if shuffle:
        X, y = sklearn.utils.shuffle(X, y)

    num_batches = X.shape[0] // batch_size
    for i in range(num_batches):
        idx = i * batch_size
        yield X[idx:idx+batch_size], y[idx:idx+batch_size]

if __name__ == '__main__':
    DIM = 224

    # load data
    train, test, num_to_name = load_data('./data/101_ObjectCategories', new_size=DIM)
    X_train, y_train = train
    X_test, y_test = test

    X_train = np.asarray(X_train, dtype=np.float32) / 255.0
    X_test = np.asarray(X_test, dtype=np.float32) / 255.0

    y_train = np_utils.to_categorical(y_train, len(num_to_name))
    y_test = np_utils.to_categorical(y_test, len(num_to_name))

    # initialize model
    model = ResnetBuilder.build_resnet_18((3, DIM, DIM), len(num_to_name))
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train
    NUM_EPOCHS = 100

    print model.get_input_shape_at(0)

    datagen = ImageDataGenerator()

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = time.time()
        batches = 0
        for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=32):
            batches += 1
            model.fit(X_batch, y_batch)
            if batches >= len(X_train) / 32:
                break
        print "Training time = {:.3f}".format(time.time()-start_time)

        val_err, val_acc = model.evaluate(X_test, y_test, batch_size=32)

        print "Val acc = {:.2f}%".format(val_acc * 100)