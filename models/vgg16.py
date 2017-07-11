from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization


def build_model(input_shape=(224, 224, 3), num_classes=1000,
                weights='imagenet'):

    builtin_model = VGG16(include_top=False, weights=weights,
                          input_shape=input_shape)

    x = builtin_model.output

    # classification block
    x = Flatten(name='flatten')(x)

    x = Dense(4096, name='fc1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='a1')(x)
    x = Dropout(0.5, name='dropout1')(x)

    x = Dense(4096, name='fc2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu', name='a2')(x)
    x = Dropout(0.5, name='dropout2')(x)

    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(builtin_model.inputs, x, name='vgg16')

    return model
