from __future__ import division

import math
import keras
import numpy as np

import os
import shutil


def count_correct(predictions, true_labels, top=1):

    correct_labels = 0

    for idx in xrange(predictions.shape[0]):
        pred = predictions[idx]
        true = true_labels[idx]

        top_k = np.argpartition(pred, -top)[-top:]
        if true in top_k:
            correct_labels += 1

    return correct_labels


def evaluate(model, test_gen, ten_crop):
    """
    Evaluate a given model with a generator.
    """
    keras.backend.clear_session()

    # load model and add metrics
    model = keras.models.load_model(model)

    steps = math.ceil(test_gen.n / test_gen.batch_size)

    correct_labels = 0
    top_5_correct_labels = 0
    num_classes = model.output_shape[1]

    for X_batch, y_batch in test_gen:
        if steps == 0:
            break
        steps -= 1

        true_labels = np.argmax(y_batch, axis=1)

        if ten_crop:
            res_shape = (X_batch.shape[0], num_classes)
            res_batch = np.zeros(res_shape, dtype=X_batch.dtype)

            for idx in xrange(len(X_batch)):
                x = X_batch[idx]
                pred = np.mean(model.predict_on_batch(x), axis=0)
                res_batch[idx] = pred

            correct_labels += count_correct(res_batch, true_labels, 1)
            top_5_correct_labels += count_correct(res_batch, true_labels, 5)

        else:
            preds = model.predict_on_batch(X_batch)

            correct_labels += count_correct(preds, true_labels, 1)
            top_5_correct_labels += count_correct(preds, true_labels, 5)

    print "Top 1:", correct_labels / test_gen.n
    print "Top 5:", top_5_correct_labels / test_gen.n


def get_wrong_predictions(model, test_gen, label_names, out_dir, ten_crop):
    """Output all the wrong predictions from a generator to a folder named
    wrong_predictions in the current working directory.

    Args:
        model: a file containing a pre-trained model
        test_gen: a generator that yields data by batches. The generator should
            have these following attributes: n (number of samples)
            and batch_size.
        label_names: a list containing class names.
        out_dir: the directory that the output images will be stored.

    Returns:
        None
    """
    keras.backend.clear_session()

    model = keras.models.load_model(model)
    paths = test_gen.paths
    steps = math.ceil(test_gen.n / test_gen.batch_size)

    idx = 0
    total_wrong_predictions = 0

    for X_batch, y_batch in test_gen:
        if steps == 0:
            break
        steps -= 1

        one_hot_predictions = model.predict_on_batch(X_batch)

        for i in xrange(len(one_hot_predictions)):
            prediction_label = np.argmax(one_hot_predictions[i])
            true_label = np.argmax(y_batch[i])

            if prediction_label != true_label:
                total_wrong_predictions += 1
                prediction_name = label_names[prediction_label]
                true_name = label_names[true_label]

                img = paths[idx]

                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                file_extension = os.path.splitext(img)[1]
                dst = os.path.join(out_dir, '{}_{}_{}{}'.format(
                                                            idx,
                                                            prediction_name,
                                                            true_name,
                                                            file_extension))
                shutil.copy(img, dst)
            idx += 1

    print "Accuracy:", 1 - total_wrong_predictions / test_gen.n
