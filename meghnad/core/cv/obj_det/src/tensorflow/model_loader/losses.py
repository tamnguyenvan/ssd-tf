import tensorflow as tf
from utils.common_defs import class_header, method_header

__all__ = ['SSDLoss']


import itertools
from typing import Any, Optional

import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()


@class_header(description='''
    Focal loss function for multiclass classification with integer labels.
    This loss function generalizes multiclass softmax cross-entropy by
    introducing a hyperparameter called the *focusing parameter* that allows
    hard-to-classify examples to be penalized more heavily relative to
    easy-to-classify examples.
    See :meth:`~focal_loss.binary_focal_loss` for a description of the focal
    loss in the binary setting, as presented in the original work [1]_.
    In the multiclass setting, with integer labels :math:`y`, focal loss is
    defined as
    .. math::
        L(y, \hat{\mathbf{p}})
        = -\left(1 - \hat{p}_y\right)^\gamma \log(\hat{p}_y)
    where
    *   :math:`y \in \{0, \ldots, K - 1\}` is an integer class label (:math:`K`
        denotes the number of classes),
    *   :math:`\hat{\mathbf{p}} = (\hat{p}_0, \ldots, \hat{p}_{K-1})
        \in [0, 1]^K` is a vector representing an estimated probability
        distribution over the :math:`K` classes,
    *   :math:`\gamma` (gamma, not :math:`y`) is the *focusing parameter* that
        specifies how much higher-confidence correct predictions contribute to
        the overall loss (the higher the :math:`\gamma`, the higher the rate at
        which easy-to-classify examples are down-weighted).
    The usual multiclass softmax cross-entropy loss is recovered by setting
    :math:`\gamma = 0`.
    Parameters
    ----------
    y_true : tensor-like
        Integer class labels.
    y_pred : tensor-like
        Either probabilities or logits, depending on the `from_logits`
        parameter.
    gamma : float or tensor-like of shape (K,)
        The focusing parameter :math:`\gamma`. Higher values of `gamma` make
        easy-to-classify examples contribute less to the loss relative to
        hard-to-classify examples. Must be non-negative. This can be a
        one-dimensional tensor, in which case it specifies a focusing parameter
        for each class.
    class_weight: tensor-like of shape (K,)
        Weighting factor for each of the :math:`k` classes. If not specified,
        then all classes are weighted equally.
    from_logits : bool, optional
        Whether `y_pred` contains logits or probabilities.
    axis : int, optional
        Channel axis in the `y_pred` tensor.

    Returns
    -------
    :class:`tf.Tensor`
        The focal loss for each example.
              ''')
def sparse_categorical_focal_loss(y_true, y_pred, gamma, *,
                                  class_weight: Optional[Any] = None,
                                  from_logits: bool = False, axis: int = -1
                                  ) -> tf.Tensor:
    # Process focusing parameter
    gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)
    gamma_rank = gamma.shape.rank
    scalar_gamma = gamma_rank == 0

    # Process class weight
    if class_weight is not None:
        class_weight = tf.convert_to_tensor(class_weight,
                                            dtype=tf.dtypes.float32)

    # Process prediction tensor
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred_rank = y_pred.shape.rank
    if y_pred_rank is not None:
        axis %= y_pred_rank
        if axis != y_pred_rank - 1:
            # Put channel axis last for sparse_softmax_cross_entropy_with_logits
            perm = list(itertools.chain(range(axis),
                                        range(axis + 1, y_pred_rank), [axis]))
            y_pred = tf.transpose(y_pred, perm=perm)
    elif axis != -1:
        raise ValueError(
            f'Cannot compute sparse categorical focal loss with axis={axis} on '
            'a prediction tensor with statically unknown rank.')
    y_pred_shape = tf.shape(y_pred)

    # Process ground truth tensor
    y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
    y_true_rank = y_true.shape.rank

    if y_true_rank is None:
        raise NotImplementedError('Sparse categorical focal loss not supported '
                                  'for target/label tensors of unknown rank')

    reshape_needed = (y_true_rank is not None and y_pred_rank is not None and
                      y_pred_rank != y_true_rank + 1)
    if reshape_needed:
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])

    if from_logits:
        logits = y_pred
        probs = tf.nn.softmax(y_pred, axis=-1)
    else:
        probs = y_pred
        logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))

    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true,
        logits=logits,
    )

    y_true_rank = y_true.shape.rank
    probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)
    if not scalar_gamma:
        gamma = tf.gather(gamma, y_true, axis=0, batch_dims=y_true_rank)
    focal_modulation = (1 - probs) ** gamma
    loss = focal_modulation * xent_loss

    if class_weight is not None:
        class_weight = tf.gather(class_weight, y_true, axis=0,
                                 batch_dims=y_true_rank)
        loss *= class_weight

    if reshape_needed:
        loss = tf.reshape(loss, y_pred_shape[:-1])

    return loss


@class_header(description='''
    Focal loss function for multiclass classification with integer labels.
    This loss function generalizes multiclass softmax cross-entropy by
    introducing a hyperparameter :math:`\gamma` (gamma), called the
    *focusing parameter*, that allows hard-to-classify examples to be penalized
    more heavily relative to easy-to-classify examples.
    This class is a wrapper around
    :class:`~focal_loss.sparse_categorical_focal_loss`. See the documentation
    there for details about this loss function.
    Parameters
    ----------
    gamma : float or tensor-like of shape (K,)
        The focusing parameter :math:`\gamma`. Higher values of `gamma` make
        easy-to-classify examples contribute less to the loss relative to
        hard-to-classify examples. Must be non-negative. This can be a
        one-dimensional tensor, in which case it specifies a focusing parameter
        for each class.
    class_weight: tensor-like of shape (K,)
        Weighting factor for each of the :math:`k` classes. If not specified,
        then all classes are weighted equally.
    from_logits : bool, optional
        Whether model prediction will be logits or probabilities.
    **kwargs : keyword arguments
        Other keyword arguments for :class:`tf.keras.losses.Loss` (e.g., `name`
        or `reduction`).
    ''')
@tf.keras.utils.register_keras_serializable()
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):

    def __init__(self, gamma, class_weight: Optional[Any] = None,
                 from_logits: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.class_weight = class_weight
        self.from_logits = from_logits

    @method_header(
        description='''
        Returns the config of the layer.
        A layer config is a Python dictionary containing the configuration of a
        layer. The same layer can be re-instantiated later (without its trained
        weights) from this configuration.''',
        returns='''
        dict
            This layer's config.
        ''')
    def get_config(self):
        config = super().get_config()
        config.update(gamma=self.gamma, class_weight=self.class_weight,
                      from_logits=self.from_logits)
        return config

    @method_header(
        description='''
        Compute the per - example focal loss.
        This method simply calls
        : meth: `~focal_loss.sparse_categorical_focal_loss` with the appropriate
        arguments.
        Returns
        -------''',
        arguments='''
        y_true: tensor - like, shape(N,)
            Integer class labels.
        y_pred: tensor - like, shape(N, K)
            Either probabilities or logits, depending on the `from_logits`
            parameter.
        ''',
        returns='''
        : class: `tf.Tensor`
            The per - example focal loss. Reduction to a scalar is handled by
            this layer's
            : meth: `~focal_loss.SparseCateogiricalFocalLoss.__call__` method.
        ''')
    def call(self, y_true, y_pred):
        return sparse_categorical_focal_loss(y_true=y_true, y_pred=y_pred,
                                             class_weight=self.class_weight,
                                             gamma=self.gamma,
                                             from_logits=self.from_logits)


@method_header(
    description=""" Hard negative mining algorithm
        to pick up negative examples for back - propagation
        base on classification loss values
    Args:
        loss: list of classification losses of all default boxes(B, num_default)
        gt_confs: classification targets(B, num_default)
        neg_ratio: negative / positive ratio
    Returns:
        conf_loss: classification loss
        loc_loss: regression loss
    """)
def hard_negative_mining(loss, gt_confs, neg_ratio):

    pos_idx = gt_confs > 0
    num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.int32), axis=1)
    num_neg = num_pos * neg_ratio

    rank = tf.argsort(loss, axis=1, direction='DESCENDING')
    rank = tf.argsort(rank, axis=1)
    neg_idx = rank < tf.expand_dims(num_neg, 1)

    return pos_idx, neg_idx


@class_header(
    description=""" Class for SSD Losses
    Attributes:
        neg_ratio: negative / positive ratio
        num_classes: number of classes
        focal_loss: Use focal loss or not .
    """)
class SSDLoss(object):

    def __init__(self, neg_ratio, num_classes, focal_loss: bool = False):
        self.neg_ratio = neg_ratio
        self.num_classes = num_classes
        self.focal_loss = focal_loss

    @method_header(
        description=""" Compute losses for SSD
            regression loss: smooth L1
            classification loss: cross entropy
        Args:
            confs: outputs of classification heads (B, num_default, num_classes)
            locs: outputs of regression heads (B, num_default, 4)
            gt_confs: classification targets (B, num_default)
            gt_locs: regression targets (B, num_default, 4)
        Returns:
            conf_loss: classification loss
            loc_loss: regression loss
        """)
    def __call__(self, confs, locs, gt_confs, gt_locs):

        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        # compute classification losses without reduction
        temp_loss = cross_entropy(
            gt_confs, confs)
        pos_idx, neg_idx = hard_negative_mining(
            temp_loss, gt_confs, self.neg_ratio)

        # classification loss will consist of positive and negative examples

        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='sum')
        smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')

        if self.focal_loss:
            focal_loss_fn = SparseCategoricalFocalLoss(2.0, from_logits=True)
            conf_loss = focal_loss_fn(
                gt_confs[tf.math.logical_or(pos_idx, neg_idx)],
                confs[tf.math.logical_or(pos_idx, neg_idx)])
        else:
            conf_loss = cross_entropy(
                gt_confs[tf.math.logical_or(pos_idx, neg_idx)],
                confs[tf.math.logical_or(pos_idx, neg_idx)])

        # regression loss only consist of positive examples
        loc_loss = smooth_l1_loss(
            gt_locs[pos_idx],
            locs[pos_idx])

        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.float32))

        conf_loss = conf_loss / num_pos
        loc_loss = loc_loss / num_pos

        return conf_loss, loc_loss
