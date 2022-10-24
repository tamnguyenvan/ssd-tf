import os
import sys
import glob
import numpy as np
import tensorflow as tf
from utils import ret_values
from utils.log import Log
from .loader_utils import get_tfrecord_dataset

log = Log()


class DataLoader:
    def __init__(self,
                 batch_size=1,
                 img_size=(600, 600, 3),
                 label_encoder=None,
                 train_test_val_split=(0.7, 0.2, 0.1)):
        self.batch_size = batch_size
        self.img_size = img_size
        self.label_encoder = label_encoder
        self.train_test_val_split = train_test_val_split
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def resize_and_pad_image(
        self, image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
    ):
        image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        if jitter is not None:
            min_side = tf.random.uniform(
                (), jitter[0], jitter[1], dtype=tf.float32)
        ratio = min_side / tf.reduce_min(image_shape)
        if ratio * tf.reduce_max(image_shape) > max_side:
            ratio = max_side / tf.reduce_max(image_shape)
        image_shape = ratio * image_shape
        image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
        padded_image_shape = tf.cast(
            tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
        )
        image = tf.image.pad_to_bounding_box(
            image, 0, 0, padded_image_shape[0], padded_image_shape[1]
        )
        return image, image_shape, ratio

    def _parse_tf_example(self, tf_example):
        """_summary_

        Parameters
        ----------
        tf_example : _type_
            _description_
        """
        example_fmt = {
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
        image = tf.image.decode_image(
            parsed_example['image/encoded'], channels=3)
        image_height = parsed_example['image/height']
        image_width = parsed_example['image/width']
        image = tf.reshape(image, (image_height, image_width, 3))

        xmins = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'])
        xmaxs = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'])
        ymins = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'])
        ymaxs = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'])
        classes = tf.cast(tf.sparse.to_dense(
            parsed_example['image/object/class/label']), tf.int32)

        image, image_shape, _ = self.resize_and_pad_image(
            image
        )

        bboxes = tf.stack([
            xmins * image_shape[1],
            ymins * image_shape[0],
            (xmaxs - xmins) * image_shape[1],
            (ymaxs - ymins) * image_shape[0]
        ])
        return image, bboxes, classes

    def load_data_from_directory(self, path, augment=False,
                                 rescale=True, rand_flip=False, rotate=False):
        self.config_connectors(path)
        autotune = tf.data.AUTOTUNE

        # Training set
        train_dataset = self.read_data(self.connector['trn_data_path'],
                                       self.connector['trn_file_path'],
                                       dataset='train')
        train_dataset = train_dataset.shuffle(8 * self.batch_size)
        train_dataset = train_dataset.map(
            self._parse_tf_example, num_parallel_calls=autotune,
        )
        train_dataset = train_dataset.padded_batch(
            batch_size=self.batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
        )
        train_dataset = train_dataset.map(
            self.label_encoder.encode_batch, num_parallel_calls=autotune
        )
        train_dataset = train_dataset.apply(
            tf.data.experimental.ignore_errors())
        self.train_dataset = train_dataset.prefetch(autotune)

        # Validation set
        validation_dataset = self.read_data(self.connector['val_data_path'],
                                            self.connector['val_file_path'],
                                            dataset='val')
        validation_dataset = validation_dataset.map(
            self._parse_tf_example, num_parallel_calls=autotune,
        )
        validation_dataset = validation_dataset.padded_batch(
            batch_size=self.batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
        )
        validation_dataset = validation_dataset.map(
            self.label_encoder.encode_batch, num_parallel_calls=autotune
        )
        validation_dataset = validation_dataset.apply(
            tf.data.experimental.ignore_errors())
        self.validation_dataset = validation_dataset.prefetch(autotune)

        # Testing set
        test_dataset = self.read_data(self.connector['test_data_path'],
                                      self.connector['test_file_path'],
                                      dataset='test')
        test_dataset = test_dataset.map(
            self._parse_tf_example, num_parallel_calls=autotune,
        )
        test_dataset = test_dataset.padded_batch(
            batch_size=self.batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
        )
        test_dataset = test_dataset.map(
            self.label_encoder.encode_batch, num_parallel_calls=autotune
        )
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
        dataset = get_tfrecord_dataset(
            image_dir, annotation_file, tfrecord_file)
        # except:
        #     return None
        return dataset

    def read_pred_data(self, path):
        images = []
        for img in glob.glob(path + "/*.png"):
            img = tf.keras.preprocessing.image.load_img(
                img, target_size=self.img_size)
            img_arr = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_arr)
        images = np.array(images)
        dataset = tf.data.Dataset.from_tensors((images))
        print(tf.data.experimental.cardinality(dataset))
        return dataset
