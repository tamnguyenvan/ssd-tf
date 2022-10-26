import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D


def get_backbone(name, image_size=(300, 300)):
    if name == 'MobileNetV2':
        return [
            tf.keras.applications.MobileNetV2(
                input_shape=(image_size[0], image_size[1], 3), include_top=False),
            ('block_13_expand_relu', 'out_relu')
        ]


def create_extra_layers(backbone):
    if backbone == 'MobileNetV2':
        extra_layers = [
            Sequential([
                Conv2D(256, (1, 1), strides=(1, 1), padding="valid",
                       activation="relu", name="extra1_1"),
                Conv2D(512, (3, 3), strides=(2, 2), padding="same",
                       activation="relu", name="extra1_2"),
            ]),
            Sequential([
                Conv2D(128, (1, 1), strides=(1, 1), padding="valid",
                       activation="relu", name="extra2_1"),
                Conv2D(256, (3, 3), strides=(2, 2), padding="same",
                       activation="relu", name="extra2_2")
            ]),
            Sequential([
                Conv2D(128, (1, 1), strides=(1, 1), padding="valid",
                       activation="relu", name="extra3_1"),
                Conv2D(256, (3, 3), strides=(2, 2), padding="same",
                       activation="relu", name="extra3_2")
            ]),
            Sequential([
                Conv2D(128, (1, 1), strides=(1, 1), padding="valid",
                       activation="relu", name="extra4_1"),
                Conv2D(256, (3, 3), strides=(2, 2), padding="same",
                       activation="relu", name="extra4_2")
            ])
        ]

        return extra_layers


def create_heads(backbone, num_classes, num_anchors):
    if backbone == 'MobileNetV2':
        conf_head_layers = []
        loc_head_layers = []
        for i in range(len(num_anchors)):
            conf_head_layers.append(
                Conv2D(num_anchors[i] * num_classes,
                       kernel_size=3, strides=1, padding='same')
            )
            loc_head_layers.append(
                Conv2D(num_anchors[i] * 4,
                       kernel_size=3, strides=1, padding='same')
            )
        return conf_head_layers, loc_head_layers
