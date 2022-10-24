#######################################################################################################################
# Configurations for natural language translator.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Mayank Jain
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.cfg.config import MeghnadConfig

import sys

log = Log()

_translator_cfg =\
{
    'max_chars': 5000
}    

class LangTransConfig(MeghnadConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.threshold = _translator_cfg['max_chars']  

    def get_max_chars(self) -> int:
        return self.threshold

   
