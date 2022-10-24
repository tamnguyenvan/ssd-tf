#######################################################################################################################
# Configurations for NLP emotion detector.
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

_emotion_cfg =\
{
    'dims': ['joy', 'love', 'excitement', 'admiration', 'approval', 'pride', 'caring', 'amusement', 'realization',
             'desire', 'relief', 'curiosity', 'surprise', 'optimism', 'gratitude', 'neutral', 'remorse', 'nervousness',
             'annoyance', 'anger', 'grief', 'fear', 'disapproval', 'confusion', 'embarrassment', 'disgust', 'sadness',
             'disappointment',],
    'lang': 'en',
    'hypothesis_template': 'The emotion for this text is {}.',
}

class DetectEmotionConfig(MeghnadConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_emotion_dims(self) -> [[]]:
        try:
            return _emotion_cfg['dims'].copy()
        except:
            return _emotion_cfg['dims']

    def get_emotion_lang(self) -> str:
        try:
            return _emotion_cfg['lang'].copy()
        except:
            return _emotion_cfg['lang']

    def get_emotion_hypothesis_template(self) -> str:
        try:
            return _emotion_cfg['hypothesis_template'].copy()
        except:
            return _emotion_cfg['hypothesis_template']

