#######################################################################################################################
# Configurations for Topic model.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Chiranjeevraja
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.cfg.config import MeghnadConfig


log = Log()

# Default Parameters
_topic_model_cfg = \
{
    'stop_words': 'english',  # Language of the documents
    'max_features' : 20000,   # Maximum number of words to be trained
    'n_hidden' : 50,          # Number of topics
    'anchor_strength' : 6     # Setting anchor strength greater than 5 is strongly enforcing that the CorEx \
                                                # topic model find a topic associated with the anchor words
}

class TopicModelConfig():
    def __init__(self):
        pass
    def get_config(self):
        return _topic_model_cfg.copy()

if __name__ == '__main__':
    pass
