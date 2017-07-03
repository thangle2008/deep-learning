from __future__ import division

import random
import numpy as np

import keras
from keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau, CSVLogger,
                             EarlyStopping)

from models.resnet import ResnetBuilder
from models import vgg16


def run(model, train, val, opt, batch_size=32, num_epochs=100, **kwargs):
    """
    Train a classifier with the provided training and validation data.
    The model must be a compiled keras model.
    """
    # set the random seed for image processing and shuffling
    random.seed(28)

    # extract metadata
    num_train_samples = train.n
    num_val_samples = val.n
    num_classes = len(train.label_names)
    dim = train.output_shape[0]

    # callbacks list
    model_checkpoint = ModelCheckpoint(
        'weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(
        factor=np.sqrt(0.1), patience=5, verbose=1)
    early_stop = EarlyStopping(min_delta=0.001, patience=11)
    csv_logger = CSVLogger('{}_{}.csv'.format(model, num_classes))

    # load model
    print "Load model..."
    
    if model == 'resnet':
        model = ResnetBuilder.build_resnet((3, dim, dim), num_classes, **kwargs)
    elif model == 'vgg16':
        model = vgg16.build_model((dim, dim, 3), num_classes)

    if opt['algo'] == 'sgd':
        optimizer = keras.optimizers.SGD(**opt['params'])
    else:
        optimizer = keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print "Number of parameters =", model.count_params()
    print "Training samples =", num_train_samples
    print "Number of classes =", num_classes
    print "Training parameters =", optimizer.__class__, optimizer.get_config()

    model.fit_generator(
        train,
        steps_per_epoch=num_train_samples // batch_size,
        epochs=num_epochs,
        validation_data=val,
        validation_steps=num_val_samples // batch_size,
        callbacks=[model_checkpoint, reduce_lr, early_stop, csv_logger],
    )

    print "Best validation loss = {:.3f}".format(model_checkpoint.best)
    return model_checkpoint.best
