from __future__ import division

import math
import keras
import numpy as np

import os
import shutil


def count_correct(predictions, true_labels, top=1):
    """Count the number of correct predictions.

    Both predictions and true labels should be one-hot.

    Args:
        predictions: A Numpy array of predictions.
        true_labels: A Numpy array of labels.
        top: The least rank of the prediction to be counted as correct.

    Returns: Number of correct predictions.

    """

    correct_labels = 0

    for idx in xrange(predictions.shape[0]):
        pred = predictions[idx]
        true = true_labels[idx]

        top_k = np.argpartition(pred, -top)[-top:]
        if true in top_k:
            correct_labels += 1

    return correct_labels


def _output_wrong_labels(paths, output_dir, batch_idx, predictions,
                         true_labels):
    """Output incorrectly classified images into a folder."""

    for idx in xrange(predictions.shape[0]):
        pred = predictions[idx].argmax()
        true = true_labels[idx]

        if pred != true:
            # create the directory if it does not exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            current_idx = idx + batch_idx

            src = paths[current_idx]
            file_extension = os.path.splitext(src)[1]
            dst = os.path.join(output_dir, "{}_{}_{}{}".format(current_idx,
                                                               pred, true,
                                                               file_extension))

            shutil.copy(src, dst)


def evaluate(model, test_gen, ten_crop, output_dir=None):
    """Evaluate a trained model.

    Print out top-1 and top-5 accuracies.

    Args:
        model: A Keras model.
        test_gen: A generator that yields data by batches indefinitely.
            The generator should have 2 attributes: n (number of samples)
            and batch_size (size of each batch).
        ten_crop (bool): If true, use the ten crop method while testing.
        output_dir (str): If provided, output all the incorrectly classified
            images into the specified folder.

    """
    keras.backend.clear_session()

    # load model and add metrics
    model = keras.models.load_model(model)

    steps = math.ceil(test_gen.n / test_gen.batch_size)

    correct_labels = 0
    top_5_correct_labels = 0
    num_classes = model.output_shape[1]
    batch_idx = 0

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

            if output_dir is not None:
                _output_wrong_labels(test_gen.paths, output_dir, batch_idx,
                                     res_batch, true_labels)

        else:
            preds = model.predict_on_batch(X_batch)

            correct_labels += count_correct(preds, true_labels, 1)
            top_5_correct_labels += count_correct(preds, true_labels, 5)

            if output_dir is not None:
                _output_wrong_labels(test_gen.paths, output_dir, batch_idx,
                                     preds, true_labels)

        batch_idx += test_gen.batch_size

    print "Top 1:", correct_labels / test_gen.n
    print "Top 5:", top_5_correct_labels / test_gen.n

