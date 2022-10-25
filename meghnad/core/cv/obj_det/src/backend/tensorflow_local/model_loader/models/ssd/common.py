import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D


def get_backbone(name, input_size=300):
    if name == 'mobilenetv2':
        return [
            tf.keras.applications.MobileNet(include_top=False),
            ('block_13_expand_relu', 'out_relu')
        ]


def create_extra_layers(backbone):
    extra_layers = [
        Sequential([
            Conv2D(256, (1, 1), strides=(1, 1), padding="valid",
                   activation="relu", name="extra1_1"),
            Conv2D(512, (3, 3), strides=(2, 2), padding="same",
                   activation="relu", name="extra1_2"),
        ]),
        Sequential([
            Conv2D(128, (1, 1), strides=(1, 1), padding="valid",
                   activation="relu", name="extra2_1")
            Conv2D(256, (3, 3), strides=(2, 2), padding="same",
                   activation="relu", name="extra2_2")
        ]),
        Sequential([
            Conv2D(128, (1, 1), strides=(1, 1), padding="valid",
                   activation="relu", name="extra3_1")
            Conv2D(256, (3, 3), strides=(2, 2), padding="same",
                   activation="relu", name="extra3_2")
        ]),
        Sequential([
            Conv2D(128, (1, 1), strides=(1, 1), padding="valid",
                   activation="relu", name="extra4_1")
            Conv2D(256, (3, 3), strides=(2, 2), padding="same",
                   activation="relu", name="extra4_2")
        ])
    ]
    return extra_layers
