import os
import sys
import tempfile
import json

import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2

from utils import ret_values
from utils.log import Log
from meghnad.core.cv.obj_det.src.tensorflow.inference.vis_utils import draw_bboxes
from meghnad.core.cv.obj_det.src.tensorflow.model_loader.ssd.anchors import generate_default_boxes
from meghnad.core.cv.obj_det.src.tensorflow.model_loader.ssd.utils.ssd_box_utils import decode, compute_nms

log = Log()


class TfObjDetEval:
    def __init__(self,
                 model_loader,
                 model_config,
                 data_loader,
                 ckpt_path=None,
                 phase='test',
                 score_threshold=0.4,
                 nms_threshold=0.5,
                 max_predictions=100,
                 image_out_dir = '',
                 draw_predictions = False,
                 from_hub = False):
        self.model_loader = model_loader
        self.data_loader = data_loader

        print('Number of classes:', data_loader.num_classes)
        self.num_classes = data_loader.num_classes
        self.class_map = data_loader.class_map
        self.ckpt_path = ckpt_path
        self.dataset = data_loader.validation_dataset if phase == 'validation' else data_loader.test_dataset
        self.phase = phase
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_predictions = max_predictions
        self.image_out_dir = image_out_dir
        self.draw_predictions = draw_predictions
        self.default_boxes = generate_default_boxes(
            model_config['scales'],
            model_config['feature_map_sizes'],
            model_config['aspect_ratios']
        )

        if ckpt_path and not from_hub:
            ckpt = tf.train.Checkpoint(model=self.model_loader.model)
            ckpt.restore(ckpt_path)
            print(f'Loaded pretrained weights from {ckpt_path}')

    def eval(self):
        if self.model_loader.model is None:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Model is not fitted yet")
            return ret_values.IXO_RET_INVALID_INPUTS

        results = {'annotations': []}
        ann_id = 0
        for batch_image_ids, batch_image_shapes, batch_images, _, _ in self.dataset:
            batch_confs, batch_locs = self.model_loader.model(
                batch_images, training=False)
            for image, image_id, image_shape, confs, locs in zip(
                batch_images, batch_image_ids, batch_image_shapes, batch_confs, batch_locs
            ):

                image_id = int(image_id)
                image_id_str = str(image_id)
                image_height, image_width = image_shape.numpy()[:2]

                confs = tf.math.softmax(confs, axis=-1)
                classes = tf.math.argmax(confs, axis=-1)
                scores = tf.math.reduce_max(confs, axis=-1)

                boxes = decode(self.default_boxes, locs)

                out_boxes = []
                out_labels = []
                out_scores = []

                for c in range(1, self.num_classes):
                    cls_scores = confs[:, c]

                    score_idx = cls_scores > self.score_threshold
                    cls_boxes = boxes[score_idx]
                    cls_scores = cls_scores[score_idx]

                    nms_idx = compute_nms(cls_boxes, cls_scores, self.nms_threshold, self.max_predictions)
                    cls_boxes = tf.gather(cls_boxes, nms_idx)
                    cls_scores = tf.gather(cls_scores, nms_idx)
                    cls_labels = [c] * cls_boxes.shape[0]

                    out_boxes.append(cls_boxes)
                    out_labels.extend(cls_labels)
                    out_scores.append(cls_scores)

                out_boxes = tf.concat(out_boxes, axis=0)
                out_scores = tf.concat(out_scores, axis=0)

                boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
                boxes_resized = boxes * np.array([[*self.data_loader.img_size*2]]).astype(np.float32)
                boxes_resized = boxes_resized.astype(np.int32).tolist()
                boxes = boxes * np.array([[image_width, image_height, image_width, image_height]]).astype(np.float32)
                boxes = boxes.astype(np.int32).tolist()
                classes = np.array(out_labels)
                scores = out_scores.numpy()

                if (self.draw_predictions):
                    dest_path = os.path.join(self.image_out_dir, image_id_str+'.jpg')
                    image *= 255
                    pred_bbox_image = draw_bboxes(image[...,::-1].numpy(), boxes_resized, classes, scores, self.class_map)
                    cv2.imwrite(dest_path, pred_bbox_image)

                for box, cls, score in zip(boxes, classes, scores):
                    x1, y1, x2, y2 = box
                    ann_id += 1
                    results['annotations'].append({
                        'id': ann_id,
                        'image_id': image_id,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'area': (x2 - x1) * (y2 - y1),
                        'category_id': int(cls),
                        'score': float(score)
                    })

        ann_json = None
        if self.phase == 'validation':
            ann_json = self.data_loader.connector['val_file_path']
        elif self.phase == 'test':
            ann_json = self.data_loader.connector['test_file_path']
        else:
            raise FileNotFoundError(
                'Not found ground truth annotation file')

        pred_file = tempfile.NamedTemporaryFile('wt').name
        with open(pred_file, 'wt') as f:
            json.dump(results, f)

        gt = COCO(ann_json)  # init annotations api
        pred = COCO(pred_file)
        eval = COCOeval(gt, pred, 'bbox')
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]
        if os.path.isfile(pred_file):
            os.remove(pred_file)
        return map, map50
