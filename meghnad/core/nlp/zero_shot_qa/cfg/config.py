#######################################################################################################################
# Configurations for NLP zero-shot question-answer.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.cfg.config import MeghnadConfig

import sys

log = Log()

_zsqa_cfg =\
{
    'models':
    {
        'distilbert-base-cased-distilled-squad':
        {
            'ckpt': 'distilbert-base-cased-distilled-squad',
            'size': 261,
            'lang': 'en',
            'cased': True,
        },
        'distilbert-base-uncased-distilled-squad':
        {
            'ckpt': 'distilbert-base-uncased-distilled-squad',
            'size': 265,
            'lang': 'en',
            'cased': False,
        },
        'minilm-uncased-squad2':
        {
            'ckpt': 'deepset/minilm-uncased-squad2',
            'size': 133,
            'lang': 'en',
            'cased': False,
        },
        'roberta-base-squad2':
        {
            'ckpt': 'deepset/roberta-base-squad2',
            'size': 496,
            'lang': 'en',
            'cased': False,
        },
        'xlm-roberta-base-squad2':
        {
            'ckpt': 'deepset/xlm-roberta-base-squad2',
            'size': 1110,
            'lang': 'multi',
            'cased': True,
        },
        'xlm-roberta-large-squad2':
        {
            'ckpt': 'deepset/xlm-roberta-large-squad2',
            'size': 2240,
            'lang': 'multi',
            'cased': True,
        },
    },
    'search_models':
    {
        'multi-qa-MiniLM-L6-cos-v1':
        {
            'ckpt': 'multi-qa-MiniLM-L6-cos-v1',
            'size': 91,
            'lang': 'multi',
            'cased': False,
        },
    },
    'settings':
    {
        'default_model':
        {
            'en': 'roberta-base-squad2',
            'multi': 'xlm-roberta-base-squad2',
        },
        'default_search_model':
        {
            'multi': 'multi-qa-MiniLM-L6-cos-v1',
        },
    },
}

class ZeroShotQAConfig(MeghnadConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()

        if 'mode' in kwargs and kwargs['mode'] == 'semantic_search':
            self.semantic_search = True
            try:
                self.zsqa_models = _zsqa_cfg['search_models'].copy()
            except:
                self.zsqa_models = _zsqa_cfg['search_models']
        else:
            self.semantic_search = False
            try:
                self.zsqa_models = _zsqa_cfg['models'].copy()
            except:
                self.zsqa_models = _zsqa_cfg['models']

    def get_zsqa_model(self, model_name:str=None, lang:str='en') -> dict:
        if model_name and model_name in self.zsqa_models:
            try:
                return self.zsqa_models[model_name].copy()
            except:
                return self.zsqa_models[model_name]
        elif self.semantic_search:
            default_model = self.get_zsqa_settings('default_search_model')['multi']

            try:
                return self.zsqa_models[default_model].copy()
            except:
                return self.zsqa_models[default_model]
        else:
            if lang != 'en':
                lang = 'multi'
            default_model = self.get_zsqa_settings('default_model')[lang]

            try:
                return self.zsqa_models[default_model].copy()
            except:
                return self.zsqa_models[default_model]

    def get_zsqa_settings(self, key:str) -> str:
        if key and key in _zsqa_cfg['settings']:
            try:
                return _zsqa_cfg['settings'][key].copy()
            except:
                return _zsqa_cfg['settings'][key]

