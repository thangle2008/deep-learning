# Adapt from https://github.com/raghakot/keras-resnet
# add support for shortcut option A and smaller Resnet architecture
# for training CIFAR10 based on the original Deep Residual Learning paper

from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Lambda
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


# global variables
SHORTCUT_OPTION = 'B'
ROW_AXIS = 0
COL_AXIS = 1
CHANNEL_AXIS = 2


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    
    # if shape is different. 
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        if SHORTCUT_OPTION == 'B':
            # 1x1 convolution to match dimension
            shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(0.0001))(input)
        elif SHORTCUT_OPTION == 'A':
            # spatial pooling with padded identity mapping
            x = AveragePooling2D(pool_size=(1, 1),
                                 strides=(stride_width, stride_height))(input)
            # multiply every element of x by 0 to get zero matrix
            mul_zero = Lambda(lambda val: val * 0.0,
                              output_shape=K.int_shape(x)[1:])(x)

            shortcut = concatenate([x, mul_zero], axis=CHANNEL_AXIS)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters,
                                   init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0)
                                   )(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_shortcut_option(option='B'):
    global SHORTCUT_OPTION

    if option not in ['A', 'B']:
        raise ValueError
    else:
        SHORTCUT_OPTION = option


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, base_filters=64,
              shortcut_option='B', downsampling_top=True):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows,
                nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or
                `bottleneck`. The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the
                input size is halved
            base_filters: The number of filters that the first residual block has.
            shortcut_option: The shortcut option to use in the original paper.
                Either 'A' (identity map with padded zeros) or 'B' (convolutional
                map).
            downsampling_top: Whether to include the max pooling after the first
                convolutional layer (that layer also has stride of 2 if this
                is set to True)

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        _handle_shortcut_option(shortcut_option)

        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, "
                            "nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)

        # set up first layer
        if downsampling_top:
            # This is based on the original Resnet for tinyimagenet
            conv1 = _conv_bn_relu(filters=base_filters, kernel_size=(7, 7),
                                  strides=(2, 2))(input)
            pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                 padding="same")(conv1)
            block = pool1
        else:
            # This is based on the Resnet for Cifar10, which does not contain
            # the pooling layer
            conv1 = _conv_bn_relu(filters=base_filters, kernel_size=(3, 3),
                                  strides=(1, 1))(input)
            block = conv1

        # add residual blocks
        filters = base_filters
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, 
                                    is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS],
                                            block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet(input_shape, num_outputs, depth=18, shortcut_option='B',
                     base_filters=64, downsampling_top=True):
        """Build a Resnet model based on the given configuration.

        Args:
            input_shape (tuple): A 3-D tuple. The color channel must be in the
                first axis. For example, (3, 224, 244) is a valid tuple.
            num_outputs (int): The number of classes.
            depth (int):
            shortcut_option (str): The option for handling the case where the
                output of a residual block has different shape from that of
                the shortcut. This should be either `A` (identity mapping with
                zero padding) or `B` (convolutional mapping that matches
                two different dimensions).
            base_filters (int): The number of filters in the first residual
                block.
            downsampling_top (bool): If True, the first convolutional layer
                will have 7x7 kernels and a max pooling layer to reduce
                the dimension of the input image. Otherwise, there is only
                one convolutional layer with 3x3 kernels at the top.

        Returns: A Keras model.

        """
        params = locals()

        # number of residual blocks for each depth
        num_blocks = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            # for cifar 10
            20: [3, 3, 3],
            32: [5, 5, 5],
            56: [9, 9, 9],
            110: [18, 18, 18]
        }

        if depth not in num_blocks.keys():
            raise ValueError("Depth is not valid.")

        print "Build Resnet with ", params

        block_function = basic_block
        if depth > 34 and shortcut_option != 'A':
            block_function = bottleneck

        return ResnetBuilder.build(input_shape, num_outputs,
                                   block_function, num_blocks[depth],
                                   base_filters=base_filters,
                                   shortcut_option=shortcut_option,
                                   downsampling_top=downsampling_top)
