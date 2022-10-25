import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras import layers
from .common import get_backbone, create_extra_layers, create_heads


def ssd(backbone, input_shape, num_classes):
    input_size = input_shape[0]
    base_model, feature_names = get_backbone(backbone, input_size)
    extra_layers = create_extra_layers()

    features = []
    for name in feature_names:
        features.append(base_model.get_layer(name).output)

    x = base_model.output
    for layer in extra_layers:
        x = layer(x)
        features.append(x)

    confs = []
    locs = []
    conf_head_layers, loc_head_layers = create_heads(backbone, num_classes)
    for i, feature in enumerate(features):
        conf = conf_head_layers[i](feature)
        loc = loc_head_layers[i][feature]

        confs.append(tf.reshape(conf, [conf.shape[0], -1, 4]))
        locs.append(tf.reshape(loc, [loc.shape[0], -1, 4]))
    return Model(input, [confs, locs])
