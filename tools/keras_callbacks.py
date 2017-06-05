from keras.callbacks import Callback

import numpy as np

class BestModelCheck(Callback):
    """Keep track of the best model with lowest validation loss."""


    def __init__(self, save_path):
        super(BestModelCheck, self).__init__()
        self.save_path = save_path

    def on_train_begin(self, logs={}):
        self.best_val_loss = None
        self.best_val_acc = None


    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')
        if self.best_val_loss is None or self.best_val_loss > val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = logs.get('val_acc')
            self.model.save_weights(self.save_path)