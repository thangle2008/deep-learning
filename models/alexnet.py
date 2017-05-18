from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

def build_model(n_classes, batch_size=32, batch_norm=False):
    model = Sequential()
    model.add(Dense(input_shape=(batch_size, 227, 227, 3)))

    # conv block 1
    model.add(Conv2D(96, 11, strides=4))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, 2))

    # conv block 2
    model.add(Conv2D(256, 5, strides=4))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, 2))

    # conv block 1
    model.add(Conv2D(96, 11, strides=4))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(3, 2))