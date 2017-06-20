import keras

def evaluate(model, test_gen, data_size):
    """
    Evaluate a given model with a generator that yields one sample with 
    its true label at a time.
    """

    # load model and add metrics

    model = keras.models.load_model(model)
    model.compile(model.optimizer, model.loss, 
        metrics=['accuracy', keras.metrics.top_k_categorical_accuracy])

    print model.evaluate_generator(test_gen, data_size)