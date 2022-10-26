import os
import json

import numpy as np
import tensorflow as tf
import tqdm


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def build_example(image_dir, image_info):

    image_path = os.path.join(image_dir, image_info['filename'])
    image = open(image_path, 'rb').read()

    ext = image_info['filename'].split('.')[-1]
    if ext.lower() in ('jpg', 'jpeg'):
        ext = b'jpg'
    elif ext.lower() == 'png':
        ext = b'png'

    bboxes = np.array(image_info['bboxes'])
    xmins = bboxes[..., 0].tolist()
    xmaxs = bboxes[..., 1].tolist()
    ymins = bboxes[..., 2].tolist()
    ymaxs = bboxes[..., 3].tolist()
    class_names = [name.encode('utf-8') for name in image_info['class_names']]
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/id': _int64_feature(image_info['id']),
        'image/height': _int64_feature(image_info['height']),
        'image/width': _int64_feature(image_info['width']),
        'image/filename': _bytes_feature(image_info['filename'].encode('utf-8')),
        'image/encoded': _bytes_feature(image),
        'image/format': _bytes_feature(ext),
        'image/object/bbox/xmin': _float_list_feature(xmins),
        'image/object/bbox/xmax': _float_list_feature(xmaxs),
        'image/object/bbox/ymin': _float_list_feature(ymins),
        'image/object/bbox/ymax': _float_list_feature(ymaxs),
        'image/object/class/text': _bytes_list_feature(class_names),
        'image/object/class/label': _int64_list_feature(image_info['classes']),
    }))
    return tf_example


def get_tfrecord_dataset(image_dir, ann_file, tfrecord_file=None):
    if tfrecord_file is None:
        tfrecord_file = 'sample.tfrecord'

    if not os.path.isfile(tfrecord_file):
        with open(ann_file) as f:
            json_data = json.load(f)

        image_data = json_data['images']
        ann_data = json_data['annotations']
        categories = json_data['categories']

        images_dict = dict()
        for im in image_data:
            images_dict[im['id']] = im

        categories_dict = dict()
        for cat in categories:
            categories_dict[cat['id']] = cat

        data = dict()
        for ann in ann_data:
            image_id = ann['image_id']
            if image_id not in data:
                data[image_id] = dict()
                info = images_dict[image_id]
                image_height = info['height']
                image_width = info['width']
                data[image_id] = {
                    'id': image_id,
                    'filename': info['file_name'],
                    'height': image_height,
                    'width': image_width,
                    'bboxes': [],
                    'class_names': [],
                    'classes': []
                }

            category_id = ann['category_id']
            class_name = categories_dict[category_id]['name']

            xmin, ymin, w, h = ann['bbox']
            xmax = xmin + w
            ymax = ymin + h

            # Normalize
            image_h = data[image_id]['height']
            image_w = data[image_id]['width']
            xmin /= image_w
            ymin /= image_h
            xmax /= image_w
            ymax /= image_h

            data[image_id]['bboxes'].append((xmin, xmax, ymin, ymax))
            data[image_id]['class_names'].append(class_name)
            data[image_id]['classes'].append(category_id)

        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            for _, image_info in tqdm.tqdm(data.items()):
                example = build_example(image_dir, image_info)
                writer.write(example.SerializeToString())

    # Initialize a dataset from the above tfreocrd file
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    return dataset
