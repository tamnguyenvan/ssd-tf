from typing import Tuple
import sys

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, ReLU

from utils import ret_values
from utils.log import Log

log = Log()


def get_backbone(name, image_size=(300, 300)) -> Tuple:
    """Get backbone network from given name.

    Returns  a 3-tuple represents a model instance, a list of feature names, and
    name of the output layer.

    Parameters
    ----------
    name : str
        Model name.
    image_size : tuple, optional
        Input image size, by default (300, 300)

    Returns
    -------
    Tuple
        _description_
    """
    if name == 'MobileNetV2':
        return [
            tf.keras.applications.MobileNetV2(
                input_shape=(image_size[0], image_size[1], 3), include_top=False),
            ('block_13_expand_relu', 'out_relu'),
            'out_relu'
        ]
    elif name == 'EfficientNetB3':
        return [
            tf.keras.applications.EfficientNetB3(
                input_shape=(image_size[0], image_size[1], 3), include_top=False),
            ('block7b_add',),
            'block7b_add'
        ]
    elif name == 'EfficientNetB4':
        return [
            tf.keras.applications.EfficientNetB4(
                input_shape=(image_size[0], image_size[1], 3), include_top=False),
            ('block7b_add',),
            'block7b_add'
        ]
    elif name == 'EfficientNetB5':
        return [
            tf.keras.applications.EfficientNetB5(
                input_shape=(image_size[0], image_size[1], 3), include_top=False),
            ('block7c_add',),
            'block7c_add'
        ]
    elif name == 'EfficientNetV2S':
        return [
            tf.keras.applications.EfficientNetV2S(
                input_shape=(image_size[0], image_size[1], 3), include_top=False),
            ('block6o_add',),
            'block6o_add'
        ]
    elif name == 'EfficientNetV2M':
        return [
            tf.keras.applications.EfficientNetV2M(
                input_shape=(image_size[0], image_size[1], 3), include_top=False),
            ('block7e_add',),
            'block7e_add'
        ]
    elif name == 'EfficientNetV2L':
        return [
            tf.keras.applications.EfficientNetV2L(
                input_shape=(image_size[0], image_size[1], 3), include_top=False),
            ('block7g_add',),
            'block7g_add'
        ]
    else:
        log.ERROR(sys._getframe().f_lineno,
                  __file__, __name__, f"Unsupported backbone {name}")
        return ret_values.IXO_RET_NOT_SUPPORTED


def create_extra_layers(backbone):
    if backbone == 'MobileNetV2':
        extra_layers = [
            Sequential([
                Conv2D(256, (1, 1), strides=(1, 1), padding="valid",
                       activation=ReLU(6.0), name="extra1_1"),
                Conv2D(512, (3, 3), strides=(2, 2), padding="same",
                       activation=ReLU(6.0), name="extra1_2"),
            ]),
            Sequential([
                Conv2D(128, (1, 1), strides=(1, 1), padding="valid",
                       activation=ReLU(6.0), name="extra2_1"),
                Conv2D(256, (3, 3), strides=(2, 2), padding="same",
                       activation=ReLU(6.0), name="extra2_2")
            ]),
            Sequential([
                Conv2D(128, (1, 1), strides=(1, 1), padding="valid",
                       activation=ReLU(6.0), name="extra3_1"),
                Conv2D(256, (3, 3), strides=(2, 2), padding="same",
                       activation=ReLU(6.0), name="extra3_2")
            ]),
            Sequential([
                Conv2D(128, (1, 1), strides=(1, 1), padding="valid",
                       activation=ReLU(6.0), name="extra4_1"),
                Conv2D(256, (3, 3), strides=(2, 2), padding="same",
                       activation=ReLU(6.0), name="extra4_2")
            ])
        ]

        return extra_layers
    elif backbone == 'EfficientNetB3':
        extra_layers = [
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                MaxPooling2D((2, 2))
            ])
        ]
        return extra_layers
    elif backbone == 'EfficientNetB4':
        extra_layers = [
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                MaxPooling2D((2, 2))
            ])
        ]
        return extra_layers
    elif backbone == 'EfficientNetB5':
        extra_layers = [
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                MaxPooling2D((2, 2))
            ])
        ]
        return extra_layers
    elif backbone == 'EfficientNetV2S':
        extra_layers = [
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                MaxPooling2D((2, 2))
            ])
        ]
        return extra_layers
    elif backbone == 'EfficientNetV2M':
        extra_layers = [
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                MaxPooling2D((2, 2))
            ])
        ]
        return extra_layers
    elif backbone == 'EfficientNetV2L':
        extra_layers = [
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(320, (3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(),
                ReLU(6.0),
                MaxPooling2D((2, 2))
            ]),
            Sequential([
                MaxPooling2D((2, 2))
            ])
        ]
        return extra_layers
    else:
        log.ERROR(sys._getframe().f_lineno,
                  __file__, __name__, f"Unsupported backbone {backbone}")
        return ret_values.IXO_RET_NOT_SUPPORTED


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
    elif backbone == 'EfficientNetB3':
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
    elif backbone == 'EfficientNetV2S':
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


def build_fpn(feat_size):
    return Sequential([
        Conv2D(64, feat_size)
    ])