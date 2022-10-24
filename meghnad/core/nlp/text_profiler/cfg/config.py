#######################################################################################################################
# Configurations for text profiler.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.cfg.config import MeghnadConfig

import sys, string, nltk

log = Log()

_text_profiler_cfg =\
{
    'common_puncts': [",", ".", "'", "!", '"', ";", "?", "-", ":", ";"],
    'functional_words': ['a', 'between', 'in', 'nor', 'some', 'upon', 'about', 'both', 'including', 'nothing',
                         'somebody', 'us', 'above', 'but', 'inside', 'of', 'someone', 'used', 'after', 'by', 'into',
                         'off', 'something', 'via', 'all', 'can', 'is', 'on', 'such', 'we', 'although', 'cos', 'it',
                         'once', 'than', 'what', 'am', 'do', 'its', 'one', 'that', 'whatever', 'among', 'down',
                         'latter', 'onto', 'the', 'when', 'an', 'each', 'less', 'opposite', 'their', 'where', 'and',
                         'either', 'like', 'or', 'them', 'whether', 'another', 'enough', 'little', 'our', 'these',
                         'which', 'any', 'every', 'lots', 'outside', 'they', 'while', 'anybody', 'everybody', 'many',
                         'over', 'this', 'who', 'anyone', 'everyone', 'me', 'own', 'those', 'whoever', 'anything',
                         'everything', 'more', 'past', 'though', 'whom', 'are', 'few', 'most', 'per', 'through',
                         'whose', 'around', 'following', 'much', 'plenty', 'till', 'will', 'as', 'for', 'must',
                         'plus', 'to', 'with', 'at', 'from', 'my', 'regarding', 'toward', 'within', 'be', 'have',
                         'near', 'same', 'towards', 'without', 'because', 'he', 'need', 'several', 'under', 'worth',
                         'before', 'her', 'neither', 'she', 'unless', 'would', 'behind', 'him', 'no', 'should',
                         'unlike', 'yes', 'below', 'i', 'nobody', 'since', 'until', 'you', 'beside', 'if', 'none',
                         'so', 'up', 'your'],
    'min_chars_in_para': 3,
    'key_phrase':
    {
        'nr_candidates': 20,
        'rake_min_score': 0.25,
        'key_bert_min_score': 0.25,
        'top_n_as_pct_of_seq_len': 0.25,
        'pos_filter': [],
    },
}

class TextProfilerConfig(MeghnadConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._set_text_profiler_cfg()

    def _set_text_profiler_cfg(self):
        _text_profiler_cfg['special_chars'] = list(set(list(string.punctuation))\
                                                   - set(_text_profiler_cfg['common_puncts']))
        _text_profiler_cfg['stop_words'] = nltk.corpus.stopwords.words('english') + list(string.punctuation)
        _text_profiler_cfg['functional_words'] = list(set(_text_profiler_cfg['functional_words'])\
                                                      - set(_text_profiler_cfg['stop_words']))

    def get_text_profiler_cfg(self) -> dict:
        try:
            return _text_profiler_cfg.copy()
        except:
            return _text_profiler_cfg

