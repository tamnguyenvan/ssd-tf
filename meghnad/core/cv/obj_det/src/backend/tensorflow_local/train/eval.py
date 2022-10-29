import sys

import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import ret_values
from utils.log import Log
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.models.ssd.anchors import generate_default_boxes
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.models.ssd.utils.box_utils import decode, compute_nms

log = Log()


class ModelEvaluator:
    def __init__(self,
                 model_loader=None,
                 model_config=None,
                 data_loader=None,
                 weights=None,
                 phase='test'):
        self.model_loader = model_loader
        self.data_loader = data_loader
        self.weights = weights
        self.dataset = data_loader.validation_dataset if phase == 'validation' else data_loader.test_dataset
        self.phase = phase
        self.default_boxes = generate_default_boxes(
            model_config['scales'],
            model_config['feature_map_sizes'],
            model_config['aspect_ratios']
        )

        if weights:
            self.model_loader.model.load_weights(weights)
            print(f'Loaded pretrained weights from {weights}')

    def eval(self):
        if self.model_loader.model is None:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Model is not fitted yet")
            return ret_values.IXO_RET_INVALID_INPUTS
        # if len(self.dataset.cardinality().shape) == 0:
        #     log.ERROR(sys._getframe().f_lineno,
        #               __file__, __name__, "Test data is empty")
        #     return ret_values.IXO_RET_INVALID_INPUTS

        results = []
        for batch_image_ids, batch_image_shapes, batch_images, _, _ in self.dataset:
            batch_confs, batch_locs = self.model_loader.model(batch_images, training=False)
            for image_id, image_shape, confs, locs in zip(
                batch_image_ids, batch_image_shapes, batch_confs, batch_locs
            ):
                image_id = int(image_id)
                image_height, image_width = image_shape.numpy()[:2]

                confs = tf.math.softmax(confs, axis=-1)
                classes = tf.math.argmax(confs, axis=-1)
                scores = tf.math.reduce_max(confs, axis=-1)

                boxes = decode(self.default_boxes, locs)

                out_boxes = []
                out_labels = []
                out_scores = []

                for c in range(1, 3):
                    cls_scores = confs[:, c]

                    score_idx = cls_scores > 0.6
                    # cls_boxes = tf.boolean_mask(boxes, score_idx)
                    # cls_scores = tf.boolean_mask(cls_scores, score_idx)
                    cls_boxes = boxes[score_idx]
                    cls_scores = cls_scores[score_idx]

                    nms_idx = compute_nms(cls_boxes, cls_scores, 0.45, 200)
                    cls_boxes = tf.gather(cls_boxes, nms_idx)
                    cls_scores = tf.gather(cls_scores, nms_idx)
                    cls_labels = [c] * cls_boxes.shape[0]

                    out_boxes.append(cls_boxes)
                    out_labels.extend(cls_labels)
                    out_scores.append(cls_scores)

                out_boxes = tf.concat(out_boxes, axis=0)
                out_scores = tf.concat(out_scores, axis=0)

                boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
                classes = np.array(out_labels)
                scores = out_scores.numpy()

                for box, cls, score in zip(boxes, classes, scores):
                    x1, y1, x2, y2 = box
                    result = {
                        'image_id': image_id,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'category_id': cls,
                        'score': score
                    }
                    results.append(result)

        print('=======================', len(results))
        if len(results):
            ann_json = None
            if self.phase == 'validation':
                ann_json = self.data_loader.connector['val_file_path']
            elif self.phase == 'test':
                ann_json = self.data_loader.connector['test_file_path']
            else:
                raise FileNotFoundError('Not found ground truth annotation file')

            anno = COCO(ann_json)  # init annotations api
            pred = anno.loadRes(results)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]
        else:
            print('No valid validation results')
            map, map50 = 0.0, 0.0
        return map, map50

        # return ret_values.IXO_RET_SUCCESS, loss, metrics
