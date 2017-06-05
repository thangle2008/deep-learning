# adapt from https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
import keras
import keras.backend as K
from keras.models import Model
from keras.layers.core import Flatten, Dense


def build_model(include_top=True, weights='imagenet',
                input_tensor=None, input_shape=None,
                pooling=None,
                classes=1000):

    builtin_model = keras.applications.vgg16.VGG16(
        include_top=False, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape, 
        pooling=None, classes=classes)

    x = builtin_model.outputs[0]

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    model = Model(builtin_model.inputs, x, name='vgg16')

    return model