from __future__ import division

import math
import keras

def evaluate(model, test_gen):
    """
    Evaluate a given model with a generator.
    """

    # load model and add metrics

    model = keras.models.load_model(model)
    model.compile(model.optimizer, model.loss, 
        metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])

    steps = math.ceil(test_gen.n / test_gen.batch_size)
    print model.evaluate_generator(test_gen, steps)