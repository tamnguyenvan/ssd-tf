import os
import json
import sys
import glob
import numpy as np
import tensorflow as tf
from utils import ret_values
from utils.log import Log
from .loader_utils import get_tfrecord_dataset
from ..model_loader.models.ssd.anchors import generate_default_boxes
from ..model_loader.models.ssd.utils.box_utils import compute_target

log = Log()


class DataLoader:
    def __init__(self,
                 num_classes,
                 batch_size=4,
                 img_size=(300, 300),
                 scales=[0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],
                 feature_map_sizes=[38, 19, 10, 5, 3, 1],
                 aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_boxes = 100
        # self.label_encoder = label_encoder
        self.train_dataset = None
        self.train_size = 0
        self.validation_dataset = None
        self.val_size = 0
        self.test_dataset = None
        self.test_size = 0

        self.default_boxes = generate_default_boxes(
            scales, feature_map_sizes, aspect_ratios)

    def _parse_tf_example(self, tf_example, training=True):
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
        image = tf.cast(tf.image.resize(image, self.img_size), tf.float32)
        image /= 255.0

        xmins = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'])
        ymins = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'])
        xmaxs = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'])
        ymaxs = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'])
        labels = tf.cast(tf.sparse.to_dense(
            parsed_example['image/object/class/label']), tf.int32)
        num_pad = tf.maximum(0, self.max_boxes - tf.shape(labels)[0])
        labels = tf.pad(labels, [[0, num_pad]])

        bboxes = tf.stack([
            xmins,
            ymins,
            xmaxs,
            ymaxs,
        ], 1)
        bboxes = tf.pad(bboxes, [[0, num_pad], [0, 0]])
        bboxes = tf.reshape(bboxes, [self.max_boxes, 4])
        labels = tf.reshape(labels, [self.max_boxes])
        gt_confs, gt_locs = compute_target(
            self.default_boxes, bboxes, labels)
        if training:
            return image, gt_confs, gt_locs
        else:
            return image_id, tf.stack([image_height, image_width]), image, gt_confs, gt_locs

    def load_data_from_directory(self, path, augment=False,
                                 rescale=True, rand_flip=False, rotate=False):
        self.config_connectors(path)
        autotune = tf.data.AUTOTUNE

        # Load class map
        with open(self.connector['test_file_path']) as f:
            test_ann_data = json.load(f)
            self.class_map = {cate['id']: cate['name'] for cate in test_ann_data['categories']}

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
        # train_dataset = train_dataset.map(
        #     self.label_encoder.encode_batch, num_parallel_calls=autotune
        # )
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
        # validation_dataset = validation_dataset.map(
        #     self.label_encoder.encode_batch, num_parallel_calls=autotune
        # )
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
        # test_dataset = test_dataset.map(
        #     self.label_encoder.encode_batch, num_parallel_calls=autotune
        # )
        test_dataset = test_dataset.apply(
            tf.data.experimental.ignore_errors())
        self.test_dataset = test_dataset.prefetch(autotune)
        if augment:
            self.augment_data(
                rescale=rescale, random_flip=rand_flip, random_rotation=rotate)
        return self.train_dataset, self.validation_dataset, self.test_dataset

    def augment_data(self, rescale=True, random_flip=False, random_rotation=False):

        data_augmentation_layers = tf.keras.Sequential()
        if rescale:
            data_augmentation_layers.add(
                tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255))
        if random_flip:
            data_augmentation_layers.add(
                tf.keras.layers.RandomFlip('horizontal'))
        if random_rotation:
            data_augmentation_layers.add(tf.keras.layers.RandomRotation(0.2))
        if not self.train_dataset:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Training data is null")
            return ret_values.IXO_RET_INVALID_INPUTS
        if not self.validation_dataset:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Validation data is null")
            return ret_values.IXO_RET_INVALID_INPUTS
        if not self.test_dataset:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Test data is null")
            return ret_values.IXO_RET_INVALID_INPUTS
        self.train_dataset = (
            self.train_dataset
                .shuffle(self.batch_size * 100)
                .batch(self.batch_size)
                .map(lambda x, y: (data_augmentation_layers(x), y),
                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .prefetch(tf.data.experimental.AUTOTUNE)
        )
        self.test_dataset = (
            self.test_dataset
                .shuffle(self.batch_size * 100)
                .batch(self.batch_size)
                .map(lambda x, y: (data_augmentation_layers(x), y),
                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .prefetch(tf.data.experimental.AUTOTUNE)
        )
        self.validation_dataset = (
            self.validation_dataset
                .shuffle(self.batch_size * 100)
                .batch(self.batch_size)
                .map(lambda x, y: (data_augmentation_layers(x), y),
                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .prefetch(tf.data.experimental.AUTOTUNE)
        )

        return ret_values.IXO_RET_SUCCESS

    def config_connectors(self, path: str):
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

    def load_data_from_url(
        self,
        url,
        save_as=None,
        data_dir=None,
        augment=False,
        rescale=True,
        rand_flip=False,
        rotate=False
    ):
        """_summary_

        Parameters
        ----------
        url : _type_
            _description_
        save_as : _type_, optional
            _description_, by default None
        data_dir : _type_, optional
            _description_, by default None
        augment : bool, optional
            _description_, by default False
        rescale : bool, optional
            _description_, by default True
        rand_flip : bool, optional
            _description_, by default False
        rotate : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        path_to_downloaded_dataset = tf.keras.utils.get_file(
            save_as, origin=url, extract=True, cache_dir=data_dir)
        base_path = os.path.join(os.path.dirname(
            path_to_downloaded_dataset), save_as.partition(".")[0])
        if not os.path.exists(base_path):
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Invalid path for data: {} ".format(base_path))
            return ret_values.IXO_RET_INVALID_INPUTS

        train_dir_full_path = os.path.join(base_path, 'train')
        if os.path.exists(train_dir_full_path):
            self.train_dir = train_dir_full_path
        else:
            self.train_dir = base_path

        test_dir_full_path = os.path.join(base_path, 'test')
        if os.path.exists(test_dir_full_path):
            self.test_dir = test_dir_full_path
        else:
            self.test_dir = None
        val_dir_full_path = os.path.join(base_path, 'validation')
        if os.path.exists(val_dir_full_path):
            self.val_dir = val_dir_full_path
        else:
            self.val_dir = None
        if augment:
            self.augment_data(
                rescale=rescale, random_flip=rand_flip, random_rotation=rotate)
        return ret_values.IXO_RET_SUCCESS, self.train_dir, self.test_dir, self.val_dir

    def read_data(self, image_dir, annotation_file, dataset='train'):
        # try:
        tfrecord_file = f'{dataset}.tfrecord'
        dataset, num_samples = get_tfrecord_dataset(
            image_dir, annotation_file, tfrecord_file)
        # except:
        #     return None
        return dataset, num_samples

    def read_pred_data(self, path):
        images = []
        for img in glob.glob(path + "/*.png"):
            img = tf.keras.preprocessing.image.load_img(
                img, target_size=self.img_size)
            img_arr = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_arr)
        images = np.array(images)
        dataset = tf.data.Dataset.from_tensors((images))
        return dataset
