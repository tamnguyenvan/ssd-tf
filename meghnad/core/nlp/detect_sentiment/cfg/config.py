#######################################################################################################################
# Configurations for sentiment detection.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Avinash Kumar Yadav
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.cfg.config import MeghnadConfig

import sys

log = Log()

_sentiment_cfg =\
{
    'neutral_threshold_score' : 0.6,
    'sentiment_labels' : ['Positive','Negative','Neutral'],
}    

class SentimentConfig(MeghnadConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_cutoff_score(self) -> int:
        return _sentiment_cfg['neutral_threshold_score']

    def get_sentiment_labels(self) -> list:
        try:
            return _sentiment_cfg['sentiment_labels'].copy()
        except:
            return _sentiment_cfg['sentiment_labels']

