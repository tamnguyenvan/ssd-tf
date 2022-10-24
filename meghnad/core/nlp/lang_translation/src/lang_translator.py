#######################################################################################################################
# Language Translator for natural language.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Mayank Jain
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from connectors.interfaces.interface import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.lang_translation.cfg.config import LangTransConfig
from deep_translator import GoogleTranslator
import sys

log = Log()

@class_header(
description = '''
Language Translator for NLP.''')
class LangTranslator():
    def __init__(self, *args, **kwargs):
       if "source" in kwargs:
            source = kwargs["source"]
       else:
            source = "auto"
       if "target" in kwargs:
            target = kwargs["target"]
       else:
            target = "en"
       self.config = LangTransConfig()
       self.model, self.max_chars = _load_model(self.config, source = source, target = target)
       
    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}

    @method_header(
        description = '''
        Translate text to english from any other language text provided.''',
        arguments = '''
        sequence: text input for which the translation needs to be done
        source: source language, either given while calling or code will predict itself.
        target: by default as english (en)''',
        returns = '''
        a translated sequence of text in english''')

    def translator(self, sequence:str, source:str = 'auto', target:str = 'en') -> (str, list):
        sequence = str(sequence)
        if not self.model:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "State machine corrupted. Resetting the state machine and defaulting to auto")
            self.model, self.max_chars = _load_model(self.config, 'auto', 'en')
            source = 'auto'
            target = 'en'
        if len(sequence) < self.max_chars:
            response = self.model.translate(sequence)
        else:
            log.STATUS(sys._getframe().f_lineno,
                      __file__, __name__,"More than  5000  characters.")
            to_be_translate_text = sequence
            list_of_texts = to_be_translate_text.replace(".", "").split()
            list_of_trans_texts = []
            for i in list_of_texts:
                translated_text = self.model.translate(i)
                list_of_trans_texts.append(translated_text)
            response = ""
            for i in list_of_trans_texts:
                response = str(response) + ' '+ str(i) 

        log.VERBOSE(sys._getframe().f_lineno,
                    __file__, __name__,
                    "Sequence: {}, Language: {}, Result: {}".format(sequence, source, response))

        return response

# Load appropriate model
def _load_model(config:object, source:str = 'auto', target:str = 'en') -> (str, object):
    max_chars = config.get_max_chars()
    source = str(source)
    try:
        model = GoogleTranslator(source = source, target = target)
    except:
        model = None

    return model, max_chars


if __name__ == '__main__':
    pass

