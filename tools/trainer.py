from __future__ import division

import numpy as np

from keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau, CSVLogger)


def run(model, train, val, num_epochs=100, lr_plateau=np.sqrt(0.1), patience=5):
    """Train a Keras model with given training and validation data.

    The learning rate will be reduced if the validation loss does not get better
    in a given number of epochs.

    At the end of the training process, there will be 2 output files:
        * best_model.hdf5 (stores the model with lowest validation loss)
        * training.csv (the training log in csv format)

    Args:
        model: a Keras model (Sequential or Model)
        train: a generator that yields data by batches indefinitely. The
            generator must have 2 attributes, n (the number of samples) and
            batch_size (the size of each batch). You can examine the source
            code of class DirectoryDataGenerator in module utils.datagen to
            have an idea of how to make one (or you can look for
            ImageDataGenerator source code of Keras).
        val: same as the train generator. This data will be used for validating
            the classifier accuracy.
        num_epochs: The number of training epochs.
        lr_plateau: The factor that the current learning rate will be reduced
            if the validation loss does not get better.
        patience: The number of epochs to wait before reducing learning rate.

    Returns: The best validation loss.

    """
    # extract metadata
    num_train_samples = train.n
    num_val_samples = val.n
    train_batch_size = train.batch_size
    val_batch_size = val.batch_size
    optimizer = model.optimizer

    # callbacks list
    model_checkpoint = ModelCheckpoint(
        'best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(
        factor=lr_plateau, patience=patience, verbose=1)
    csv_logger = CSVLogger('training.csv')

    print "Number of parameters =", model.count_params()
    print "Training samples =", num_train_samples
    print "Number of classes =", model.output_shape[1]
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


# def optimize_params(model, train, val, num_outputs, dim, num_epochs=10,
#                     **kwargs):
#     """
#     Optimize hyperparameters of a given training model.
#     """
#
#     from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
#
#     space = {
#         'lr': hp.choice('lr', [0.001, 0.01, 0.1]),
#         'nesterov': True,
#         'momentum': 0.9
#     }
#
#     def objective(params):
#         opt = {
#             'algo': 'sgd',
#             'params': params
#         }
#         score = trainer.run(model, train, val, opt, num_outputs, dim,
#                             num_epochs=num_epochs, **kwargs)
#         return {'loss': score, 'status': STATUS_OK}
#
#     trials = Trials()
#
#     best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)
#     return best
