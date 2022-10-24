#######################################################################################################################
# Prediction from text classifier.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
####################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from connectors.interfaces.interface import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.text_clf.cfg.config import TextClfConfig
from meghnad.core.nlp.text_clf.src.arch import TextClfArch

import sys
from joblib import load
import numpy as np
import pandas as pd

import tensorflow as tf

log = Log()

@class_header(
description='''
Prediction from fine-tuned text classifier.''')
class TextClfPred():
    def __init__(self, saved_model_dir:str, *args, **kwargs):
        self.configs = TextClfConfig(MeghnadConfig())

        if not saved_model_dir:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Directory where trained model was saved is mandatory input needed for prediction")
        else:
            self.model_details = _load_model_details(saved_model_dir)
            model_arch = self.model_details['model_arch']
            self.arch = TextClfArch(self.configs, model_arch['num_classes'])

            if 'model' in self.model_details:
                if model_arch['source'] == 'sklearn':
                    self.model = load(self.model_details['model'])
                else:
                    self.model = tf.keras.models.load_model(self.model_details['model'])
            elif 'model_weights' in self.model_details:
                self.model = self.arch.load_arch_details(model_arch, self.model_details['model_weights'])
            else:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__,
                          "Trained model detils not found")

    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}

    @method_header(
        description='''
            Predict label.''',
        arguments='''
            sequence: text input for which the label needs to be predicted
            multi_label [optional]: whether multiple labels can be true simultaneously
            proba_thr [optional]: min confidence threshold above which label to be predicted as True
            top_n [optional]: max number of labels to be returned (valid if multi_label is true)''',
        returns='''
            a 4 member tuple containing return value, predicted label(s), corresponding score(s), 
            and a dictionary containing all labels and their respective scores''')
    def pred(self, sequence:str, multi_label:bool=False,
             proba_thr:float=None, top_n:int=None) -> (int, [str], [float], dict):
        if not self.model_details:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Model details not found")
            return IXO_RET_INVALID_INPUTS, None, None
        else:
            model_arch = self.model_details['model_arch']

        x_test, _ = self.arch.tokenize_input(model_arch, [str(sequence)])
        if model_arch['source'] == 'sklearn':
            proba_pred = self.model.predict_proba(x_test)
            if model_arch['multi_label']:
                proba_pred = [[elem[0][1] for elem in proba_pred]]
        else:
            proba_pred = self.model.predict(x_test)

        if 'target_map' not in self.model_details:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Target map not available")
            return IXO_RET_GENERIC_FAILURE, None, None
        else:
            target_map = self.model_details['target_map']

        scores_pred = sorted(proba_pred[0], reverse=True)
        labels_pred = [target_map[idx] for idx in np.argsort(proba_pred[0])]
        labels_pred.reverse()

        pred_dict = {}
        for label, score in zip(labels_pred, scores_pred):
            pred_dict[label] = score

        if multi_label:
            if model_arch['multi_label']:
                if top_n:
                    scores_pred = scores_pred[:top_n]
                    labels_pred = labels_pred[:top_n]

                if proba_thr:
                    scores_pred = [score for score in scores_pred if score > proba_thr]
                    labels_pred = labels_pred[:len(scores_pred)]

                if 'target_cols' in self.model_details:
                    target_cols = self.model_details['target_cols']
            else:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__,
                          "Multi-label prediction is not possible for a model not trained for multi-label")
                return IXO_RET_INCORRECT_CONFIG, None, None
        else:
            scores_pred = scores_pred[0]
            labels_pred = labels_pred[0]

            if proba_thr and scores_pred <= proba_thr:
                scores_pred = []
                labels_pred = []

        return IXO_RET_SUCCESS, labels_pred, scores_pred, pred_dict

# Load model details saved during training
def _load_model_details(saved_model_dir:str) -> dict:
    model_details = {}

    if saved_model_dir:
        saved_model_details_path = saved_model_dir + 'saved_model_details.joblib'
        saved_model_details = load(saved_model_details_path)

        saved_prep_paths = saved_model_details['saved_prep_paths']
        for key in saved_prep_paths:
            model_details[key] = load(saved_prep_paths[key])

        saved_arch_paths = saved_model_details['saved_arch_paths']
        for key in saved_arch_paths:
            if key == 'model' or key == 'model_weights':
                model_details[key] = saved_arch_paths[key]
            else:
                model_details[key] = load(saved_arch_paths[key])

    return model_details

if __name__ == '__main__':
    pass

