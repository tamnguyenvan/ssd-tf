#######################################################################################################################
# Sentiment Analyser for natural language.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Avinash Yadav
#######################################################################################################################

from utils.ret_values import *
from utils.common_defs import *
from utils.log import Log
from connectors.interfaces.interface import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.detect_sentiment.cfg.config import SentimentConfig
from meghnad.core.nlp.detect_lang.src.detect_lang import DetectLang
from meghnad.core.nlp.zero_shot_clf.src.zero_shot_clf import ZeroShotClf
from meghnad.core.nlp.lang_translation.src.lang_translator import LangTranslator

from flair.models import TARSClassifier
from flair.data import Sentence

import sys

log = Log()

@class_header(
description = '''
Zero Shot  multi-language Sentiment Analyser for NLP.''')
class DetectSentiment():
    def __init__(self, *args, **kwargs):        
        if 'lang' in kwargs:
            self.lang = kwargs['lang']
            if 'hypothesis_template' in kwargs:
                hypothesis_template = kwargs['hypothesis_template']
            else:
                hypothesis_template = None
            self.model = ZeroShotClf(mode = 'sentiment', hypothesis_template = hypothesis_template, lang = self.lang)
        else:
            self.lang = None
            self.model = ZeroShotClf(mode = 'sentiment')

        self.tars = TARSClassifier.load('tars-base')
        self.detect_lang = DetectLang()
        self.translate =  LangTranslator()
        self.configs = SentimentConfig()
        self.cutoff_score = self.configs.get_cutoff_score()
        self.sentiment_labels = self.configs.get_sentiment_labels()

    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}
        
    @method_header(
        description = '''
        Predict Sentiment among the labels provided.''',
        arguments = '''
        sequence : text input for which the label needs to be predicted''',
        returns = '''
        a 3 member tuple containing predicted label (predicted sentiment label,
        corresponding scores predicted and language detected respectively'''
        )
    def pred(self, sequence:str) -> (str, dict, str):
        sequence = str(sequence)

        if self.lang == 'en':
            sequence_lang = 'en'
            tars_sequence = Sentence(sequence)
            self.tars.predict_zero_shot(tars_sequence, self.sentiment_labels, multi_label=False)
            if tars_sequence.to_dict()['labels']:
                tars_result = tars_sequence.to_dict()['labels'][0]
                if tars_result['value'].lower() == 'neutral':
                    label_pred = tars_result['value']
                    score_pred = tars_result['confidence']
                    lang = self.lang

                    score_labels = {}
                    score_labels[label_pred] = score_pred
                    for label in self.sentiment_labels:
                        if label != label_pred:
                            score_labels[label] = (1 - score_pred) / (len(self.sentiment_labels) - 1)
                    return label_pred, score_labels, lang
        else:
            sequence_lang = self.detect_lang.pred(sequence)

        if sequence_lang !=  'en':
            sequence = self.translate.translator(sequence)
            
        if len(self.sentiment_labels) < 3:
            label_pred, score_labels, lang, attributions = self.model.pred(sequence = sequence,
                                                                           candidate_labels = self.sentiment_labels)
            lang = sequence_lang
            return label_pred, score_labels, lang
        else:
            label_pred, score_labels, lang, attributions = self.model.pred(sequence = sequence,
                                                                           candidate_labels = self.sentiment_labels)
            lang = sequence_lang
            score_pred = score_labels[label_pred]
            neutral_label = self.sentiment_labels[ [index for index, label in enumerate(self.sentiment_labels)\
                                                    if label.lower() == 'neutral'][0] ]
            if score_pred < self.cutoff_score and label_pred.lower() != 'neutral':
                try:
                    score_neutral = score_labels[ neutral_label ]
                except:
                    score_neutral = self.cutoff_score

                label_pred = neutral_label
                score_adjusted = score_neutral + self.cutoff_score - score_pred
                score_labels[neutral_label] = score_adjusted
            return label_pred, score_labels, lang

if __name__ == '__main__':
    pass