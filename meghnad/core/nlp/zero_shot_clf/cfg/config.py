#######################################################################################################################
# Configurations for NLP zero-shot classifier.
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

_zsc_cfg =\
{
    'models':
    {
        'en':
        {
            'ckpt': 'microsoft/deberta-large-mnli', #'Narsil/deberta-large-mnli-zero-cls',
            'hypothesis_template': 'This text is about {}.',
            'expl_ckpt': 'all-MiniLM-L6-v2', #'facebook/bart-large-mnli',
        },
        'multi':
        {
            'ckpt': 'joeddav/xlm-roberta-large-xnli', #'vicgalle/xlm-roberta-large-xnli-anli',
            'hypothesis_template': '{}',
            'expl_ckpt': 'multi-qa-MiniLM-L6-cos-v1',
        },
    },
    'senti_models':
    {
        'en':
        {
            'ckpt': 'facebook/bart-large-mnli',
            'hypothesis_template': 'The sentiment for this review is {}.',
        },
        'multi':
        {
            'ckpt': 'joeddav/xlm-roberta-large-xnli',
            'hypothesis_template': '{}',
        },
    },
    'settings':
    {
        'multi_val_sep': ';;',
        'min_lift_over_rand': 0.1,
        'min_pct_of_topmost_conf': 0.75,
        'attributions':
        {
            'top_n_max_limit': 20,
            'top_n_max_pct_of_seq_len': 0.5,
        },
    },
}

class ZeroShotClfConfig(MeghnadConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if 'mode' in kwargs:
            self.mode = kwargs['mode']
        else:
            self.mode = None

        if self.mode == 'sentiment':
            try:
                self.zsc_models = _zsc_cfg['senti_models'].copy()
            except:
                self.zsc_models = _zsc_cfg['senti_models']
        else:
            try:
                self.zsc_models = _zsc_cfg['models'].copy()
            except:
                self.zsc_models = _zsc_cfg['models']

    def get_zsc_model(self, lang:str='en') -> dict:
        if lang in self.zsc_models:
            try:
                return self.zsc_models[lang].copy()
            except:
                return self.zsc_models[lang]
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Language code {} not supported. \
                      Use get_langs_supported() method to know \
                      the list of language codes currently supported".format(lang))

    def get_langs_supported(self) -> [str]:
        try:
            return list(self.zsc_models.keys()).copy()
        except:
            return list(self.zsc_models.keys())

    def get_zsc_settings(self, key:str) -> str:
        if key and key in _zsc_cfg['settings']:
            try:
                return _zsc_cfg['settings'][key].copy()
            except:
                return _zsc_cfg['settings'][key]

