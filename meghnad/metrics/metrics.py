#######################################################################################################################
# Calculate accuracy metrics for various predictions from Meghnad.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
####################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log

import sys
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn import metrics

log = Log()


@class_header(
    description='''
    Calculate various accuracy metrics on predictions from Meghnad's classifiers.''')
class ClfMetrics():
    def __init__(self, *args, **kwargs):
        pass

    @method_header(
        description='''
            Calculate default accuracy metrics for classification tasks.''',
        arguments='''
            y_actual: actual labels
            y_pred: predicted labels
            multi_label [optional]: whether multiple labels can be true simultaneously''',
        returns='''
            a 2 member tuple containing confusion matrix and classification report''')
    def default_metrics(self, y_actual: object, y_pred: object, multi_label: bool = False,
                        seperator: str = None) -> (object, object):
        if multi_label:
            if not isinstance(y_actual[0], list):
                y_actual = y_actual.apply(lambda x: x.split(seperator))
                y_actual = y_actual.apply(lambda x: literal_eval(str(x)))
            assert(isinstance(y_actual[0], list))
            assert(isinstance(y_pred[0], list))

            encoder = MultiLabelBinarizer()
            all_labels = pd.concat([y_actual, y_pred], axis=0)
            encoder.fit_transform(all_labels)

            encoded_labels = encoder.transform(y_actual)
            encoded_labels_pred = encoder.transform(y_pred)

            cnf_mat = multilabel_confusion_matrix(encoded_labels, encoded_labels_pred)
            clf_rep = classification_report(encoded_labels, encoded_labels_pred)
        else:
            cnf_mat = confusion_matrix(y_actual, y_pred)
            clf_rep = classification_report(y_actual, y_pred)

        return cnf_mat, clf_rep

    @method_header(
        description='''
            Calculate user specified performance metrics for classification tasks.''',
        arguments='''
            metric: performance metric which is to be calculated ['roc_auc_score', 'accuracy', 'balanced_accuracy', 
            'average_precision', 'f1', 'precision','recall'],
            y_true: actual value,
            y_pred: predicted value,
            y_prob: predicted probability.''',
        returns='''
            a 2 member tuple with IXO ret value and value of the calculated performance metric.''')
    # function to calculate the user specified performance metric
    def user_metrics(self, metric: str, y_true: [int], y_pred: [int], y_prob: [float],
                     **kwargs) -> (int, float):
        if metric == 'roc_auc_score':
            if 'multi_class' in kwargs:
                return IXO_RET_SUCCESS, metrics.roc_auc_score(y_true, y_prob, multi_class=kwargs['multi_class'])
            elif 'average' in kwargs:
                return IXO_RET_SUCCESS, metrics.roc_auc_score(y_true, y_pred, average=kwargs['average'])
            else:
                return IXO_RET_SUCCESS, metrics.roc_auc_score(y_true, y_prob)
        elif metric == 'accuracy':
            return IXO_RET_SUCCESS, metrics.accuracy_score(y_true, y_pred)
        elif metric == 'balanced_accuracy':  # Not supported for multi-label target
            try:
                return IXO_RET_SUCCESS, metrics.balanced_accuracy_score(y_true, y_pred)
            except:
                log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                          f"{metric} not supported for multi-label target. "
                          "Choose any one of ['roc_auc_score', 'accuracy', 'f1', 'precision','recall']")
                return IXO_RET_NOT_SUPPORTED, None
        elif metric == 'average_precision':  # Not supported for multi-class, multi-label target
            try:
                if 'average' in kwargs:
                    return IXO_RET_SUCCESS, metrics.average_precision_score(y_true, y_prob, average=kwargs['average'])
                else:
                    return IXO_RET_SUCCESS, metrics.average_precision_score(y_true, y_prob)
            except:
                log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                          f"{metric} not supported for multi-class, multi-label target. "
                          "Choose any one of ['roc_auc_score', 'accuracy', 'balanced_accuracy', 'f1', "
                          "'precision','recall']")
                return IXO_RET_NOT_SUPPORTED, None
        elif metric == 'f1':
            if 'average' in kwargs:
                return IXO_RET_SUCCESS, metrics.f1_score(y_true, y_pred, average=kwargs['average'])
            else:
                return IXO_RET_SUCCESS, metrics.f1_score(y_true, y_pred)
        elif metric == 'precision':
            if 'average' in kwargs:
                return IXO_RET_SUCCESS, metrics.precision_score(y_true, y_pred, average=kwargs['average'])
            else:
                return IXO_RET_SUCCESS, metrics.precision_score(y_true, y_pred)
        elif metric == 'recall':
            if 'average' in kwargs:
                return IXO_RET_SUCCESS, metrics.recall_score(y_true, y_pred, average=kwargs['average'])
            else:
                return IXO_RET_SUCCESS, metrics.recall_score(y_true, y_pred)
        else:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      f"{metric} performance metric not implemented. Choose any one of ['roc_auc_score', 'accuracy', "
                      f"'balanced_accuracy', 'average_precision', 'f1', 'precision','recall']")
            return IXO_RET_NOT_IMPLEMENTED, None


if __name__ == '__main__':
    pass
