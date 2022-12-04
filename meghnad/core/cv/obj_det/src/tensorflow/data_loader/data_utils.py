import os
import json
from typing import Dict, Callable

import numpy as np
import tensorflow as tf

from meghnad.core.cv.obj_det.src.tensorflow.data_loader import bbox_utils


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def build_record(image_dir: str, image_info: Dict) -> tf.train.Example:
    image_path = os.path.join(image_dir, image_info['filename'])
    image = open(image_path, 'rb').read()

    ext = image_info['filename'].split('.')[-1]
    if ext.lower() in ('jpg', 'jpeg'):
        ext = b'jpg'
    elif ext.lower() == 'png':
        ext = b'png'

    bboxes = np.array(image_info['bboxes'])
    ymins = bboxes[..., 0].tolist()
    xmins = bboxes[..., 1].tolist()
    ymaxs = bboxes[..., 2].tolist()
    xmaxs = bboxes[..., 3].tolist()
    class_names = [name.encode('utf-8')
                   for name in image_info['class_names']]
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/id': _int64_feature(image_info['id']),
        'image/height': _int64_feature(image_info['height']),
        'image/width': _int64_feature(image_info['width']),
        'image/filename': _bytes_feature(image_info['filename'].encode('utf-8')),
        'image/encoded': _bytes_feature(image),
        'image/format': _bytes_feature(ext),
        'image/object/difficult': _int64_list_feature(image_info['difficult']),
        'image/object/bbox/xmin': _float_list_feature(xmins),
        'image/object/bbox/xmax': _float_list_feature(xmaxs),
        'image/object/bbox/ymin': _float_list_feature(ymins),
        'image/object/bbox/ymax': _float_list_feature(ymaxs),
        'image/object/class/text': _bytes_list_feature(class_names),
        'image/object/class/label': _int64_list_feature(image_info['labels']),
    }))
    return tf_example


def parse_example(tf_example):
    example_fmt = {
        'image/id': tf.io.FixedLenFeature([], tf.int64),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/difficult': tf.io.VarLenFeature(tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    parsed_example = tf.io.parse_single_example(
        tf_example, example_fmt)
    image_id = tf.cast(parsed_example['image/id'], tf.int32)
    image = tf.image.decode_jpeg(parsed_example['image/encoded'])
    image_height = tf.cast(parsed_example['image/height'], tf.int32)
    image_width = tf.cast(parsed_example['image/width'], tf.int32)
    image = tf.reshape(image, (image_height, image_width, 3))

    xmins = tf.sparse.to_dense(
        parsed_example['image/object/bbox/xmin'])
    ymins = tf.sparse.to_dense(
        parsed_example['image/object/bbox/ymin'])
    xmaxs = tf.sparse.to_dense(
        parsed_example['image/object/bbox/xmax'])
    ymaxs = tf.sparse.to_dense(
        parsed_example['image/object/bbox/ymax'])
    labels = tf.cast(tf.sparse.to_dense(
        parsed_example['image/object/class/label']), tf.int32)

    is_difficult = tf.cast(tf.sparse.to_dense(
        parsed_example['image/object/difficult']), tf.bool)
    bboxes = tf.stack([ymins, xmins, ymaxs, xmaxs], axis=-1)
    return {'image': image, 'objects': {'bbox': bboxes, 'label': labels, 'is_difficult': is_difficult}}


def preprocessing(
        image_data: Dict,
        final_height: int,
        final_width: int,
        augmentation_fn: Callable,
        phase: str):
    img = image_data['image']
    gt_boxes = image_data['objects']['bbox']
    gt_labels = tf.cast(image_data['objects']['label'], tf.int32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    if phase != 'train':
        not_diff = tf.logical_not(image_data['objects']['is_difficult'])
        gt_boxes = gt_boxes[not_diff]
        gt_labels = gt_labels[not_diff]

    if augmentation_fn:
        img, gt_boxes, gt_labels = augmentation_fn(img, gt_boxes, gt_labels)

    return img, gt_boxes, gt_labels


def compute_targets(prior_boxes, gt_boxes, gt_labels, hyper_params):
    """Calculate ssd actual output values.
    Batch operations supported.
    inputs:
        prior_boxes = (total_prior_boxes, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_boxes (batch_size, gt_box_size, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_labels (batch_size, gt_box_size)
        hyper_params = dictionary

    outputs:
        bbox_deltas = (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
        bbox_labels = (batch_size, total_bboxes, [0,0,...,0])
    """
    batch_size = tf.shape(gt_boxes)[0]
    total_labels = hyper_params["num_classes"]
    iou_threshold = hyper_params["iou_threshold"]
    variances = hyper_params["variances"]
    total_prior_boxes = prior_boxes.shape[0]
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = bbox_utils.generate_iou_map(prior_boxes, gt_boxes)
    # Get max index value for each row
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    #
    pos_cond = tf.greater(merged_iou_map, iou_threshold)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_boxes = tf.where(tf.expand_dims(
        pos_cond, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    bbox_deltas = bbox_utils.get_deltas_from_bboxes(
        prior_boxes, expanded_gt_boxes) / variances
    #
    gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_labels = tf.where(
        pos_cond, gt_labels_map, tf.zeros_like(gt_labels_map))
    bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)
    #
    return bbox_deltas, bbox_labels


def get_data_types():
    """Generating data types for tensorflow datasets.
    outputs:
        data types = output data types for (images, ground truth boxes, ground truth labels)
    """
    return (tf.float32, tf.float32, tf.int32)


def get_data_shapes():
    """Generating data shapes for tensorflow datasets.
    outputs:
        data shapes = output data shapes for (images, ground truth boxes, ground truth labels)
    """
    return ([None, None, None], [None, None], [None, ])


def get_padding_values():
    """Generating padding values for missing values in batch for tensorflow datasets.
    outputs:
        padding values = padding values with dtypes for (images, ground truth boxes, ground truth labels)
    """
    return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
