#######################################################################################################################
# Zero-shot question-answer (extractive, closed) for natural language.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from connectors.interfaces.interface import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.zero_shot_qa.cfg.config import ZeroShotQAConfig

import sys, nltk

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

log = Log()

@class_header(
description='''
Zero-shot multi-language question-answer for NLP.''')
class ZeroShotQA():
    def __init__(self, *args, **kwargs):
        if 'mode' in kwargs:
            self.configs = ZeroShotQAConfig(MeghnadConfig(), mode=kwargs['mode'])
        else:
            self.configs = ZeroShotQAConfig(MeghnadConfig())

        if 'lang' in kwargs:
            self._load_model(kwargs['lang'])
        else:
            self._load_model()

    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}

    @method_header(
        description='''
        Predict extractive closed response to (or semantic search of) the question,
        given the sequence provided as the context.''',
        arguments='''
        sequence: text input to be considered as the context
        question: question that needs to be answered / searched semantically
        top_n [optional]: top n sentences to be returned (valid only if it was initialized with semantic_search mode)''',
        returns='''
        a dictionary containing the response''')
    def pred(self, sequence:str, question:str, top_n:int=None) -> dict:
        sequence = str(sequence)
        question = str(question)
        response = {}

        if self.configs.semantic_search:
            question_emb = self.model.encode(question)
            sents = nltk.tokenize.sent_tokenize(sequence)

            response['score'] = 0
            if top_n:
                response['top_n'] = []
            for sent_idx, sent in enumerate(sents):
                sent_emb = self.model.encode(sent)
                score = util.cos_sim(question_emb, sent_emb).numpy()[0][0]

                if top_n:
                    details = {}
                    details['score'] = score
                    details['sent_idx'] = sent_idx
                    details['sent'] = sent

                    response['top_n'].append(details)

                if score > response['score']:
                    response['score'] = score
                    response['sent_idx'] = sent_idx
                    response['sent'] = sent

            if top_n:
                response = sorted(response['top_n'], key=lambda x: x['score'], reverse=True)[:top_n]
        else:
            response = self.model(context=sequence, question=question)

        log.STATUS(sys._getframe().f_lineno,
                   __file__, __name__,
                   "Sequence: {}, Response: {}".\
                   format(sequence, response))

        return response

    # Load appropriate pipeline
    def _load_model(self, lang:str=None):
        if lang:
            model_config = self.configs.get_zsqa_model(lang=str(lang))
        else:
            model_config = self.configs.get_zsqa_model()

        if self.configs.semantic_search:
            self.model = SentenceTransformer(model_config['ckpt'])
        else:
            self.model = pipeline('question-answering',
                                  model=model_config['ckpt'],
                                  tokenizer=model_config['ckpt'])

if __name__ == '__main__':
    pass

