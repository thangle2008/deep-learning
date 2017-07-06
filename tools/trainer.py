from __future__ import division

import random
import numpy as np

import keras
from keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau, CSVLogger,
                             EarlyStopping)

from models.resnet import ResnetBuilder
from models import vgg16


def run(model, train, val, opt, num_classes, dim, num_epochs=100, **kwargs):
    """
    Train a classifier with the provided training and validation data.
    The model must be a compiled keras model.
    """
    # set the random seed for image processing and shuffling
    random.seed(28)

    # extract metadata
    num_train_samples = train.n
    num_val_samples = val.n
    train_batch_size = train.batch_size
    val_batch_size = val.batch_size

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
        weights = 'tinyimagenet' if kwargs['pretrained'] else None
        model = vgg16.build_model((dim, dim, 3), num_classes, weights=weights)

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
        steps_per_epoch=num_train_samples // train_batch_size,
        epochs=num_epochs,
        validation_data=val,
        validation_steps=num_val_samples // val_batch_size,
        callbacks=[model_checkpoint, reduce_lr, csv_logger],
    )

    print "Best validation loss = {:.3f}".format(model_checkpoint.best)
    return model_checkpoint.best
