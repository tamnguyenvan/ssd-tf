#######################################################################################################################
# Language detector for natural language.
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

import sys

import fasttext
from huggingface_hub import hf_hub_download

log = Log()

@class_header(
description='''
Language detector.''')
class DetectLang():
    def __init__(self, *args, **kwargs):
        self.configs = MeghnadConfig()
        self.cache_dir = self.configs.get_meghnad_configs('HF_HUB_PATH') + "fasttext/julien-c/"
        self.fasttext_model = fasttext.load_model(hf_hub_download("julien-c/fasttext-language-id", "lid.176.bin",
                                                                  cache_dir=self.cache_dir))

    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}

    @method_header(
        description='''
        Predict language.''',
        arguments='''
        sequence: text input for which the language needs to be predicted''',
        returns='''
        language detected''')
    def pred(self, sequence:str) -> str:
        seq_lang = 'en'
        sequence = str(sequence).replace('\n', ' ')

        try:
            # seq_lang = detect(sequence)
            seq_lang = self.fasttext_model.predict(sequence, k=1)[0][0].split("__label__")[1]
            log.VERBOSE(sys._getframe().f_lineno,
                        __file__, __name__,
                        "Language detected. Sequence: {}, Language: {}".format(sequence, str(seq_lang)))
        except:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Language detection failed! Sequence: {}".format(sequence))

        return seq_lang

if __name__ == '__main__':
    pass

