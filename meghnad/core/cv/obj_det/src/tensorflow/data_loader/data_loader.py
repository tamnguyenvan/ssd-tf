import os
import json
import tensorflow as tf
from utils import ret_values
from utils.log import Log

from .loader_utils import get_tfrecord_dataset
from ..model_loader.ssd.anchors import generate_default_boxes
from ..model_loader.ssd.utils import compute_target
from .transforms import build_transforms


__all__ = ['TFObjDetDataLoader']

log = Log()


class TFObjDetDataLoader:
    def __init__(self, data_path, data_cfg, model_cfg):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg

        self.batch_size = model_cfg.get('batch_size', 4)
        self.img_size = model_cfg['input_shape'][:2]
        self.num_classes = model_cfg['num_classes']
        self.max_boxes = 100
        scales = model_cfg.get(
            'scales', [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075])

        feature_map_sizes = model_cfg.get(
            'feature_map_sizes', [38, 19, 10, 5, 3, 1])
        aspect_ratios = model_cfg.get(
            'aspect_ratios', [[2], [2, 3], [2, 3], [2, 3], [2], [2]])

        self.train_dataset = None
        self.train_size = 0
        self.validation_dataset = None
        self.val_size = 0
        self.test_dataset = None
        self.test_size = 0

        self.train_transforms = build_transforms(
            data_cfg['augmentations']['train'])
        self.test_transforms = build_transforms(
            data_cfg['augmentations']['test'])

        self.default_boxes = generate_default_boxes(
            scales, feature_map_sizes, aspect_ratios)

        self._load_data_from_directory(data_path)

    def _aug_fn(self, fn, image: tf.Tensor, bboxes: tf.Tensor, classes: tf.Tensor):
        data = {'image': image, 'bboxes': bboxes, 'classes': classes}
        aug_data = fn(**data)
        return aug_data['image'], aug_data['bboxes'], aug_data['classes']

    def _parse_tf_example(self, tf_example, to_targets=True, training=True):
        """_summary_

        Parameters
        ----------
        tf_example : _type_
            _description_
        """
        example_fmt = {
            'image/id': tf.io.FixedLenFeature([], tf.int64),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        }
        parsed_example = tf.io.parse_single_example(tf_example, example_fmt)
        image_id = tf.cast(parsed_example['image/id'], tf.int32)
        image = tf.image.decode_jpeg(parsed_example['image/encoded'])
        image_height = tf.cast(parsed_example['image/height'], tf.int32)
        image_width = tf.cast(parsed_example['image/width'], tf.int32)
        image = tf.reshape(image, (image_height, image_width, 3))
        # image = tf.image.resize(image, self.img_size)

        xmins = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'])
        ymins = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'])
        xmaxs = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'])
        ymaxs = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'])
        labels = tf.cast(tf.sparse.to_dense(
            parsed_example['image/object/class/label']), tf.int32)

        bboxes = tf.stack([
            xmins,
            ymins,
            xmaxs,
            ymaxs,
        ], 1)

        # Transformations
        fn = self.train_transforms if training else self.test_transforms
        image, bboxes, labels = tf.numpy_function(
            func=self._aug_fn,
            inp=[fn, image, bboxes, labels],
            Tout=[tf.float32, tf.float32, tf.int32])

        # Pad
        num_pad = tf.maximum(0, self.max_boxes - tf.shape(labels)[0])
        bboxes = tf.pad(bboxes, [[0, num_pad], [0, 0]])
        labels = tf.pad(labels, [[0, num_pad]])
        bboxes = tf.reshape(bboxes, [self.max_boxes, 4])
        labels = tf.reshape(labels, [self.max_boxes])

        # Compute targets
        gt_confs, gt_locs = compute_target(
            self.default_boxes, bboxes, labels)
        if training:
            return image, gt_confs, gt_locs
        else:
            return image_id, tf.stack([image_height, image_width]), image, gt_confs, gt_locs

    def _load_data_from_directory(self, path):
        self._config_connectors(path)
        autotune = tf.data.AUTOTUNE

        # Load class map
        with open(self.connector['test_file_path']) as f:
            test_ann_data = json.load(f)
            self.class_map = {cate['id']: cate['name']
                              for cate in test_ann_data['categories']}

        # Training set
        train_dataset, self.train_size = self.read_data(self.connector['trn_data_path'],
                                                        self.connector['trn_file_path'],
                                                        dataset='train')
        train_dataset = train_dataset.shuffle(8 * self.batch_size)
        train_dataset = train_dataset.map(
            lambda x: self._parse_tf_example(x, True), num_parallel_calls=autotune,
        )
        train_dataset = train_dataset.padded_batch(
            batch_size=self.batch_size, padding_values=(0.0, 0, 0.0), drop_remainder=True
        )
        train_dataset = train_dataset.apply(
            tf.data.experimental.ignore_errors())
        self.train_dataset = train_dataset.prefetch(autotune)

        # Validation set
        validation_dataset, self.val_size = self.read_data(self.connector['val_data_path'],
                                                           self.connector['val_file_path'],
                                                           dataset='val')
        validation_dataset = validation_dataset.map(
            lambda x: self._parse_tf_example(x, False), num_parallel_calls=autotune,
        )
        validation_dataset = validation_dataset.padded_batch(
            batch_size=self.batch_size, padding_values=(0, 0, 0.0, 0, 0.0), drop_remainder=True
        )
        validation_dataset = validation_dataset.apply(
            tf.data.experimental.ignore_errors())
        self.validation_dataset = validation_dataset.prefetch(autotune)

        # Testing set
        test_dataset, self.test_size = self.read_data(self.connector['test_data_path'],
                                                      self.connector['test_file_path'],
                                                      dataset='test')
        test_dataset = test_dataset.map(
            lambda x: self._parse_tf_example(x, False), num_parallel_calls=autotune,
        )
        test_dataset = test_dataset.padded_batch(
            batch_size=self.batch_size, padding_values=(0, 0, 0.0, 0, 0.0), drop_remainder=True
        )
        test_dataset = test_dataset.apply(
            tf.data.experimental.ignore_errors())
        self.test_dataset = test_dataset.prefetch(autotune)
        return self.train_dataset, self.validation_dataset, self.test_dataset

    def _config_connectors(self, path: str):
        """Dataset supposed to be COCO format"""
        self.connector = {}
        self.connector['trn_data_path'] = os.path.join(path, 'images')
        self.connector['trn_file_path'] = os.path.join(
            path, 'train_annotations.json')
        self.connector['test_data_path'] = os.path.join(path, 'images')
        self.connector['test_file_path'] = os.path.join(
            path, 'test_annotations.json')
        self.connector['val_data_path'] = os.path.join(path, 'images')
        self.connector['val_file_path'] = os.path.join(
            path, 'val_annotations.json')

    def read_data(self, image_dir, annotation_file, dataset='train'):
        tfrecord_file = f'{dataset}.tfrecord'
        dataset, num_samples = get_tfrecord_dataset(
            image_dir, annotation_file, tfrecord_file)
        return dataset, num_samples
