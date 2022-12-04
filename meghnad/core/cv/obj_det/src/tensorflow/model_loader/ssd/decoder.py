from typing import Dict

import tensorflow as tf
from tensorflow.keras.layers import Layer
from meghnad.core.cv.obj_det.src.tensorflow.data_loader import bbox_utils


__all__ = ['SSDDecoder']


class SSDDecoder(Layer):
    """Generating bounding boxes and labels from ssd predictions.
    First calculating the boxes from predicted deltas and label probs.
    Then applied non max suppression and selecting top_n boxes by scores.
    inputs:
        pred_deltas = (batch_size, total_prior_boxes, [delta_y, delta_x, delta_h, delta_w])
        pred_label_probs = (batch_size, total_prior_boxes, [0,0,...,0])
    outputs:
        pred_bboxes = (batch_size, top_n, [y1, x1, y2, x2])
        pred_labels = (batch_size, top_n)
            1 to total label number
        pred_scores = (batch_size, top_n)
    """

    # def __init__(self, prior_boxes, variances, max_total_size=200, score_threshold=0.5, **kwargs):
    def __init__(self, cfg: Dict, **kwargs):
        super(SSDDecoder, self).__init__(**kwargs)
        self.prior_boxes = bbox_utils.generate_prior_boxes(
            cfg['feature_map_shapes'], cfg['aspect_ratios']
        )
        self.variances = cfg['variances']
        self.max_total_size = 200
        self.score_threshold = cfg['score_threshold']

    def get_config(self):
        config = super(SSDDecoder, self).get_config()
        config.update({
            "prior_boxes": self.prior_boxes.numpy(),
            "variances": self.variances,
            "max_total_size": self.max_total_size,
            "score_threshold": self.score_threshold
        })
        return config

    def call(self, inputs):
        pred_deltas = inputs[0]
        pred_label_probs = inputs[1]
        batch_size = tf.shape(pred_deltas)[0]

        pred_deltas *= self.variances
        pred_bboxes = bbox_utils.get_bboxes_from_deltas(
            self.prior_boxes, pred_deltas)

        pred_labels_map = tf.expand_dims(tf.argmax(pred_label_probs, -1), -1)
        pred_labels = tf.where(tf.not_equal(
            pred_labels_map, 0), pred_label_probs, tf.zeros_like(pred_label_probs))
        # Reshape bboxes for non max suppression
        pred_bboxes = tf.reshape(pred_bboxes, (batch_size, -1, 1, 4))

        final_bboxes, final_scores, final_labels, _ = bbox_utils.non_max_suppression(
            pred_bboxes, pred_labels,
            max_output_size_per_class=self.max_total_size,
            max_total_size=self.max_total_size,
            score_threshold=self.score_threshold)

        return final_bboxes, final_labels, final_scores
