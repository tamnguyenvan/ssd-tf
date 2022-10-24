#######################################################################################################################
# Configurations for NLP coreference resolution.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Nayan Sarkar
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.cfg.config import MeghnadConfig

log = Log()

_resolve_coref_cfg =\
{
    'tag': ['PRP$', 'POS'],
    'pos': ['NOUN', 'PROPN'],
    'allenNLPurl':'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz'
}

class ResolveCorefConfig(MeghnadConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._set_resolve_coref_cfg()

    def _set_resolve_coref_cfg(self):
        pass

    def get_resolve_coref_cfg(self) -> dict:
        try:
            return _resolve_coref_cfg.copy()
        except:
            return _resolve_coref_cfg

