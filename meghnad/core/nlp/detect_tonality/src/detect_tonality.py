#######################################################################################################################
# Tonality detector for English language.
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
from meghnad.core.nlp.detect_tonality.cfg.config import DetectTonalityConfig
from meghnad.core.nlp.zero_shot_clf.src.zero_shot_clf import ZeroShotClf

import sys

log = Log()

@class_header(
description='''
Tonality detector for NLP.''')
class DetectTonality():
    def __init__(self, *args, **kwargs):
        self.configs = DetectTonalityConfig(MeghnadConfig())
        self.zsc = ZeroShotClf()
        _ = self.zsc.set_lang(lang=self.configs.get_tonality_lang(),
                              hypothesis_template=self.configs.get_tonality_hypothesis_template())

    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}

    @method_header(
        description='''
        Predict along tonality dimensions.''',
        arguments='''
        sequence: text input for which the tonality needs to be predicted''',
        returns='''
        a 2 member tuple containing predicted labels and corresponding scores predicted for each tonality dimension''')
    def pred(self, sequence:str) -> ([str], [dict]):
        labels_pred = []
        scores_pred = []

        for candidate_labels in self.configs.get_tonality_dims():
            label, scores, _, _ = self.zsc.pred(sequence, candidate_labels)
            labels_pred.append(label)
            scores_pred.append(scores)

        return labels_pred, scores_pred

if __name__ == '__main__':
    pass

