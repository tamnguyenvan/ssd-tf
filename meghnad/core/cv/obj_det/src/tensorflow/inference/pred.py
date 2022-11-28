import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from meghnad.core.cv.obj_det.src.tensorflow.model_loader.utils import decode, compute_nms
from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header, method_header


__all__ = ['TFObjDetPred']

log = Log()


@class_header(
    description='''
    Class for Object detection predictions''')
class TFObjDetPred:
    def __init__(self,
                 saved_dir,
                 output_dir='./results'):
        self.saved_dir = saved_dir
        self.output_dir = output_dir
        self.default_boxes = np.load(
            os.path.join(saved_dir, 'default_boxes.npy'))

    @method_header(
        description='''
                Function to preprocess images''',
        arguments='''
                image : image : Pass image to the function
                ''',
        returns='''
                returns image in form of a numpy array and a tuple having height and width''')
    def _preprocess(self, image):
        input_height, input_width = self.input_shape[:2]
        h, w = image.shape[:2]
        if h != input_height or w != input_width:
            image = cv2.resize(image, (input_width, input_height))
        image = image.astype(np.float32)
        image /= 255.
        return image, (h, w)

    @method_header(
        description='''
                Predict the input''',
        arguments='''
                input: input is image to the function
                score_threshold: Score threshold value.
                nms_threshold: NMS threshold value.
                max_predictions: Maximum number of predictions per image.''')
    def predict(self,
                input,
                score_threshold: float = 0.4,
                nms_threshold: float = 0.5,
                max_predictions: int = 100,
                ):
        if not self.model:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Model is null")
            return ret_values.IXO_RET_INVALID_INPUTS

        if isinstance(input, np.ndarray):
            input, shape = self._preprocess(input)
        elif isinstance(input, str):
            image = cv2.imread(input)
            image = image[:, :, ::-1]
            input, shape = self._preprocess(input)
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Not supoorted input type")
            return ret_values.IXO_RET_NOT_SUPPORTED

        # (H, W, C) -> (N, H, W, C)
        inputs = np.expand_dims(input, 0)
        batch_confs, batch_locs = self.model(inputs, training=False)
        confs = batch_confs[0]
        locs = batch_locs[0]
        image_height, image_width = shape

        confs = tf.math.softmax(confs, axis=-1)

        boxes = decode(self.default_boxes, locs)

        out_boxes = []
        out_labels = []
        out_scores = []

        num_classes = confs.shape[1]
        for c in range(1, num_classes):
            cls_scores = confs[:, c]

            score_idx = cls_scores > score_threshold
            cls_boxes = boxes[score_idx]
            cls_scores = cls_scores[score_idx]

            nms_idx = compute_nms(cls_boxes, cls_scores,
                                  nms_threshold, max_predictions)
            cls_boxes = tf.gather(cls_boxes, nms_idx)
            cls_scores = tf.gather(cls_scores, nms_idx)
            cls_labels = [c] * cls_boxes.shape[0]

            out_boxes.append(cls_boxes)
            out_labels.extend(cls_labels)
            out_scores.append(cls_scores)

        out_boxes = tf.concat(out_boxes, axis=0)
        out_scores = tf.concat(out_scores, axis=0)

        boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
        boxes = boxes * np.array([[image_width, image_height,
                                   image_width, image_height]])
        classes = np.array(out_labels)
        scores = out_scores.numpy()

        return ret_values.IXO_RET_SUCCESS, (boxes, classes, scores)
