import os
import math
import random
from pathlib import Path
from xml.etree import ElementTree
from typing import List, Tuple, Dict, Union, Callable

import tensorflow as tf
import numpy as np

from .data_utils import (
    build_record, parse_example, preprocessing,
    get_data_shapes, get_padding_values, compute_targets
)
from . import bbox_utils
from .augmentation import buil_augmentation


class TFObjDetDataLoader:
    def __init__(self, path: str, cfg: Dict):
        self.path = path
        self.cfg = cfg

    def _get_tfrecord_dataset(
            self, image_dir: str,
            data: List[Dict],
            tfrecord_file: str,
            augmentations: Dict,
            phase: str):
        if not os.path.exists(tfrecord_file):
            with tf.io.TFRecordWriter(tfrecord_file) as writer:
                for image_info in data:
                    example = build_record(image_dir, image_info)
                    writer.write(example.SerializeToString())

        # Initialize a dataset from the above tf record file
        AUTO = tf.data.AUTOTUNE
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        dataset = dataset.map(parse_example, num_parallel_calls=AUTO)
        augmentation_fn = buil_augmentation(self.cfg, augmentations)
        img_size = self.cfg['img_size']
        dataset = dataset.map(lambda x: preprocessing(
            x, img_size, img_size, augmentation_fn, phase), num_parallel_calls=AUTO)
        return dataset


class TFObjDetCOCODataLoader(TFObjDetDataLoader):
    def __init__(self, path: str, cfg: Dict):
        super().__init__(path, cfg)

    def load(self):
        pass


class TFObjDetVOCDataLoader(TFObjDetDataLoader):
    def __init__(self, path: str, cfg: Dict):
        super().__init__(path, cfg)

        if cfg['include_background']:
            self.num_classes = 1
            self.labels = ['bg']
            self.label2id = {'bg': 0}
        else:
            self.num_classes = 0
            self.labels = []
            self.label2id = dict()

    def _load_ids(self, file_path: str) -> List[str]:
        return [x.strip() for x in open(file_path)]

    def _update_labels(self, label: str):
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.num_classes += 1
            self.labels.append(label)

    def _parse_xml(self, xml_path: Union[str, Path]) -> Dict:
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext('filename')
        size = root.find('size')
        width = int(size.findtext('width'))
        height = int(size.findtext('height'))
        data = {
            'filename': filename,
            'width': width,
            'height': height,
            'bboxes': [],
            'labels': [],
            'class_names': [],
            'difficult': []
        }

        objs = root.findall('object')
        for obj in objs:
            label = obj.findtext('name')
            if obj.findtext('difficult'):
                is_difficult = int(obj.findtext('difficult'))
            else:
                is_difficult = 0
            bbox = obj.find('bndbox')
            xmin = float(bbox.findtext('xmin')) / width
            ymin = float(bbox.findtext('ymin')) / height
            xmax = float(bbox.findtext('xmax')) / width
            ymax = float(bbox.findtext('ymax')) / height

            data['bboxes'].append([ymin, xmin, ymax, xmax])
            self._update_labels(label)
            data['labels'].append(self.label2id[label])
            data['class_names'].append(label)
            data['difficult'].append(is_difficult)
        return data

    def _prepare_data(self,
                      ann_dir: Union[str, Path],
                      image_ids: List[str]) -> List[Dict]:
        ann_dir = Path(ann_dir)
        data = []
        for id_cnt, image_id in enumerate(image_ids):
            ann_path = ann_dir / (image_id + '.xml')
            ann_data = self._parse_xml(ann_path)
            ann_data['id'] = id_cnt + 1
            data.append(ann_data)
        return data

    def _build_train_dataset(self, dataset: tf.data.TFRecordDataset, batch_size: int) -> tf.data.TFRecordDataset:
        AUTO = tf.data.AUTOTUNE
        data_shapes = get_data_shapes()
        padding_values = get_padding_values()
        prior_boxes = bbox_utils.generate_prior_boxes(
            self.cfg['feature_map_shapes'], self.cfg['aspect_ratios']
        )
        self.cfg['num_classes'] = self.num_classes
        print('Number of classes:', self.num_classes)
        print(self.label2id)

        dataset = dataset.shuffle(batch_size * 4).padded_batch(
            batch_size, padded_shapes=data_shapes, padding_values=padding_values)
        dataset = dataset.map(lambda img, gt_boxes, gt_labels: (
            img, compute_targets(prior_boxes, gt_boxes, gt_labels, self.cfg)), num_parallel_calls=AUTO)
        dataset = dataset.prefetch(AUTO)
        return dataset

    def _build_test_dataset(self, dataset: tf.data.TFRecordDataset, batch_size: int) -> tf.data.TFRecordDataset:
        data_shapes = get_data_shapes()
        padding_values = get_padding_values()
        dataset = dataset.padded_batch(
            batch_size, padded_shapes=data_shapes, padding_values=padding_values)
        return dataset

    def load(self,
             batch_size: int = 8,
             augmentations: Dict = None,
             ) -> Tuple:
        # TODO: check if they exist
        root = Path(self.path)
        ann_dir = root / 'Annotations'
        imageset_dir = root / 'ImageSets' / 'Main'
        image_dir = root / 'JPEGImages'

        train_txt_file = imageset_dir / 'train.txt'
        val_txt_file = imageset_dir / 'val.txt'
        train_ids = self._load_ids(train_txt_file)
        val_ids = self._load_ids(val_txt_file)

        # Split validation set into a new validation set and a test set
        random.seed(123)
        ratio = 0.5
        num_test = int(len(val_ids) * ratio)
        random.shuffle(val_ids)
        test_ids = val_ids[:num_test]
        val_ids = val_ids[num_test:]

        # Load data from given info
        train_data = self._prepare_data(ann_dir, train_ids)
        val_data = self._prepare_data(ann_dir, val_ids)
        test_data = self._prepare_data(ann_dir, test_ids)

        # Build tfrecord datasets
        train_dataset = self._get_tfrecord_dataset(
            image_dir, train_data, 'train.tfrecord',
            augmentations, phase='train')
        val_dataset = self._get_tfrecord_dataset(
            image_dir, val_data, 'val.tfrecord',
            augmentations, phase='val'
        )
        test_dataset = self._get_tfrecord_dataset(
            image_dir, test_data, 'test.tfrecord',
            augmentations, phase='test'
        )

        train_dataset = self._build_train_dataset(train_dataset, batch_size)
        val_dataset = self._build_test_dataset(val_dataset, batch_size)
        test_dataset = self._build_test_dataset(test_dataset, batch_size)

        # Add `steps` attribute
        setattr(train_dataset, 'steps', math.ceil(len(train_ids) / batch_size))
        setattr(val_dataset, 'steps', math.ceil(len(val_ids) / batch_size))
        setattr(test_dataset, 'steps', math.ceil(len(test_ids) / batch_size))

        return train_dataset, val_dataset, test_dataset


def build_dataloader(path: str, dataset: str, cfg: Dict):
    if dataset == 'coco':
        data_loader = TFObjDetCOCODataLoader(path, cfg)
    elif dataset == 'voc':
        data_loader = TFObjDetVOCDataLoader(path, cfg)
    else:
        raise Exception('')
    return data_loader
