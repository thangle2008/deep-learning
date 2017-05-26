from keras.callbacks import Callback

import numpy as np

class BestModelCheck(Callback):
    """Keep track of the best model with lowest validation loss."""

    def on_train_begin(self, logs={}):
        self.best_val_loss = None
        self.best_weights = None

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')
        if self.best_val_loss is None or self.best_val_loss > val_loss:
            self.best_val_loss = val_loss
            self.best_weights = self.model.get_weights()