#######################################################################################################################
# Emotion detector for English language.
#
#  Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from connectors.interfaces.interface import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.detect_emotion.cfg.config import DetectEmotionConfig
from meghnad.core.nlp.zero_shot_clf.src.zero_shot_clf import ZeroShotClf

from flair.models import TARSClassifier
from flair.data import Sentence

import sys

log = Log()

@class_header(
description='''
Emotion detector for NLP.''')
class DetectEmotion():
    def __init__(self, *args, **kwargs):
        self.configs = DetectEmotionConfig(MeghnadConfig())
        self.zsc = ZeroShotClf()
        _ = self.zsc.set_lang(lang=self.configs.get_emotion_lang(),
                              hypothesis_template=self.configs.get_emotion_hypothesis_template())
        self.tars = TARSClassifier.load('tars-base')
        self.tars.switch_to_task("GO_EMOTIONS")

    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}

    @method_header(
        description='''
        Predict along emotion dimensions.''',
        arguments='''
        sequence: text input for which the emotion needs to be predicted
        multi_label [optional]: indicates whether multiple labels can be possible simultaneously
        fast [optional]: indicates whether fast turnaround time is required (if True, multi_label is assumed False)''',
        returns='''
        a 2 member tuple containing predicted label (list of labels if multi_label is True) 
        and corresponding scores predicted for each emotion dimension''')
    def pred(self, sequence:str, multi_label:bool=True, fast:bool=False) -> ([str], [dict]):
        candidate_labels = self.configs.get_emotion_dims()

        if fast:
            tars_sequence = Sentence(sequence)
            self.tars.predict(tars_sequence, multi_label=False)
            if tars_sequence.to_dict()['labels']:
                tars_result = tars_sequence.to_dict()['labels'][0]
                label_pred = tars_result['value'].lower()
                score_pred = tars_result['confidence']

                scores_pred = {}
                scores_pred[label_pred] = score_pred
                for label in candidate_labels:
                    if label != label_pred:
                        scores_pred[label] = (1 - score_pred) / (len(candidate_labels) - 1)
                labels_pred = [label_pred]
        else:
            labels_pred, scores_pred, _, _ = self.zsc.pred(sequence, candidate_labels, multi_label)

        return labels_pred, scores_pred

if __name__ == '__main__':
    pass

