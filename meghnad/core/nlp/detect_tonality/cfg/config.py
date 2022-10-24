#######################################################################################################################
# Configurations for NLP tonality detector.
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

_tonality_cfg =\
{
    'dims':
    [
        ['formal', 'casual'],  # formality
        ['serious', 'humorous'],  # humour
        ['respectful', 'irreverent'],  # respectfulness
        ['enthusiastic', 'dispassionate'],  # zeal
        ['factual', 'fanciful'],  # realism
        ['confident', 'apprehensive'],  # confidence
    ],
    'lang': 'en',
    'hypothesis_template': 'This text is {}.',
}

class DetectTonalityConfig(MeghnadConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_tonality_dims(self) -> [[]]:
        try:
            return _tonality_cfg['dims'].copy()
        except:
            return _tonality_cfg['dims']

    def get_tonality_lang(self) -> str:
        try:
            return _tonality_cfg['lang'].copy()
        except:
            return _tonality_cfg['lang']

    def get_tonality_hypothesis_template(self) -> str:
        try:
            return _tonality_cfg['hypothesis_template'].copy()
        except:
            return _tonality_cfg['hypothesis_template']

