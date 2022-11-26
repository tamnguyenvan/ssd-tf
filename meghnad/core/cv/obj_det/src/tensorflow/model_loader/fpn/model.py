import tensorflow as tf
from tensorflow.keras import Model
from .common import get_backbone, create_extra_layers, create_heads


def fpn(backbone, input_shape, num_classes, num_anchors):
    image_size = input_shape[:2]

    base_model, feature_names, base_output_name = get_backbone(
        backbone, image_size)
    extra_layers = create_extra_layers(backbone)
    conf_head_layers, loc_head_layers = create_heads(
        backbone, num_classes, num_anchors)

    input = base_model.input

    # Gather features
    features = []
    for name in feature_names:
        features.append(base_model.get_layer(name).output)

    x = base_model.get_layer(base_output_name).output
    for layer in extra_layers:
        x = layer(x)
        features.append(x)

    # Build heads
    confs = []
    locs = []
    for i, feature in enumerate(features):
        conf = conf_head_layers[i](feature)
        loc = loc_head_layers[i](feature)

        confs.append(tf.reshape(conf, [tf.shape(conf)[0], -1, num_classes]))
        locs.append(tf.reshape(loc, [tf.shape(loc)[0], -1, 4]))

    confs = tf.concat(confs, axis=1)
    locs = tf.concat(locs, axis=1)
    return Model(input, [confs, locs])
