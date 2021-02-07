import tensorflow as tf
from tensorflow.keras import layers as L


def conv_block(layer, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name=None):
    x = L.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer="he_normal", name=name)(layer)
    x = L.BatchNormalization()(x)
    x = L.Activation(tf.nn.relu)(x)
    return x


def fully_block(layer, units, activation=tf.nn.relu, dropout=0.5):
    x = L.Dense(units)(layer)
    x = L.BatchNormalization()(x)
    x = L.Dropout(dropout)(x)
    x = L.Activation(activation)(x)
    return x


class VGG16(tf.keras.Model):
    def __init__(self, num_classes, inputs, *args, **kwargs):
        x = conv_block(inputs, filters=64, name='conv1_1_64_3x3_1')
        x = conv_block(x, filters=64, name='conv1_2_64_3x3_1')
        x = L.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool_1_2x2_2')(x)

        x = conv_block(x, filters=128, name='conv2_1_128_3x3_1')
        x = conv_block(x, filters=128, name='conv2_2_128_3x3_1')
        x = L.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool_2_2x2_2')(x)

        x = conv_block(x, filters=256, name='conv3_1_256_3x3_1')
        x = conv_block(x, filters=256, name='conv3_2_256_3x3_1')
        x = conv_block(x, filters=256, name='conv3_3_256_3x3_1')
        x = L.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool_3_2x2_2')(x)

        x = conv_block(x, filters=512, name='conv4_1_512_3x3_1')
        x = conv_block(x, filters=512, name='conv4_2_512_3x3_1')
        x = conv_block(x, filters=512, name='conv4_3_512_3x3_1')
        x = L.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool_4_2x2_2')(x)

        x = conv_block(x, filters=512, name="conv5_1_512_3x3_1")
        x = conv_block(x, filters=512, name="conv5_2_512_3x3_1")
        x = conv_block(x, filters=512, name="conv5_3_512_3x3_1")
        x = L.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_5_2x2_2")(x)

        x.shape()
        # three fc layers
        x = L.Flatten()(x)

        x = fully_block(layer=x, units=4096)
        x = fully_block(layer=x, units=4096)
        x = fully_block(layer=x, units=num_classes, activation=tf.nn.softmax)

        super(VGG16, self).__init__(inputs=inputs, outputs=x, *args, **kwargs)