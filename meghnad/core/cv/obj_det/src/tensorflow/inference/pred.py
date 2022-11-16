import os
import sys
import json

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from meghnad.core.cv.obj_det.src.tensorflow.model_loader.ssd.anchors import generate_default_boxes
from meghnad.core.cv.obj_det.src.tensorflow.model_loader.ssd.utils.ssd_box_utils import decode, compute_nms
from utils import ret_values
from utils.log import Log


log = Log()


#class NoInferenceInputError(Exception):
#    pass


#class NotSupportedTypeError(Exception):
#    pass


class TfObjDetPred:
    def __init__(self,
                 model_loader,
                 model_config,
                 prediction_postprocessing=None,
                 output_dir='results'):
        self.model_loader = model_loader
        self.model_config = model_config
        self.input_shape = model_config['input_shape']
        self.model = model_loader.model
        self.prediction_postprocessing = prediction_postprocessing
        self.output_dir = output_dir
        self.default_boxes = generate_default_boxes(
            model_config['scales'],
            model_config['feature_map_sizes'],
            model_config['aspect_ratios']
        )

    def preprocess(self, image):
        input_height, input_width = self.input_shape[:2]
        h, w = image.shape[:2]
        if h != input_height or w != input_width:
            image = cv2.resize(image, (input_width, input_height))
        image = image.astype(np.float32)
        image /= 255.
        return image, (h, w)

    def predict(self, input, history=None):
        if not self.model:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Model is null")
            return ret_values.IXO_RET_INVALID_INPUTS

        if isinstance(input, np.ndarray):
            input, shape = self.preprocess(input)
        elif isinstance(input, str):
            image = cv2.imread(input)
            image = image[:, :, ::-1]
            input, shape = self.preprocess(input)
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Not supoorted input type")
            return ret_values.IXO_RET_NOT_SUPPORTED

        inputs = np.expand_dims(input, 0)
        batch_confs, batch_locs = self.model(inputs, training=False)
        confs = batch_confs[0]
        locs = batch_locs[0]
        image_height, image_width = shape
        print('==============', shape, type(image_height))

        confs = tf.math.softmax(confs, axis=-1)

        boxes = decode(self.default_boxes, locs)

        out_boxes = []
        out_labels = []
        out_scores = []

        for c in range(1, 3):
            cls_scores = confs[:, c]

            score_idx = cls_scores > 0.6
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
        boxes = boxes * np.array([[image_width, image_height,
                                   image_width, image_height]])
        classes = np.array(out_labels)
        scores = out_scores.numpy()

        return ret_values.IXO_RET_SUCCESS, (boxes, classes, scores)

    def write_prediction(self, path):
        test_ann_file = os.path.join(path, 'test_annotations.json')
        with open(test_ann_file, 'r') as f:
            annotations = json.load(f)

        for i, val in enumerate(annotations):
            image_path = '%012d.jpg' % (val['image_id'])
            full_path = path + 'test\\' + image_path
            pred = predictions[i]
            shape = ((pred[0], pred[1]),
                     (pred[0] + pred[2], pred[1] + pred[3]))
            image = Image.open(full_path)
            draw = ImageDraw.Draw(image)
            draw.rectangle(shape, outline='red')
            if not (os.path.exists(self.output_dir)):
                os.mkdir(self.output_dir)
            image.save(self.output_dir + "\\" + image_path)
