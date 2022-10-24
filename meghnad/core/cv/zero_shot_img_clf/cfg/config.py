from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.cfg.config import MeghnadConfig

import sys

log = Log()

_zsic_cfg =\
{
    'models':
    {
        'en':
        {
            'repo_id': 'openai/clip-vit-base-patch-32',
            'hypothesis_template': 'This example is {}.',
        },
         'multi':
        {
            'repo_id': 'sentence-transformers/clip-ViT-B-32-multilingual-v1',
            'hypothesis_template': 'This example is {}.',
        },

    },
    'settings':
    {
        'multi_val_sep': ';;',
    },
}


@class_header(
description='''
Zero-shot multi-language image classifier configuration for CV.''')
class ZeroShotImgClfConfig(MeghnadConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if 'mode' in kwargs:
            self.mode = kwargs['mode']
        else:
            self.mode = None

        self.zsic_models = _zsic_cfg['models']
    
    @method_header(
        description=
        '''
        Get the zsic model loaded corresponding to the language code selected
        ''')
    def get_zsic_model(self, lang:str='en') -> dict:
        if lang in self.zsic_models:
             
            try:
                return self.zsic_models[lang].copy()
            except:
                return self.zsic_models[lang]
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Language code {} not supported. \
                      Use get_langs_supported() method to know \
                      the list of language codes currently supported".format(lang))

    
    
        
    @method_header(
        description=
        '''
        Get the configuration settings for ZSIC
        ''')
    def get_zsic_settings(self, key:str) -> str:
        if key and key in _zsic_cfg['settings']:
            
            try:
                return self._zsic_cfg['settings'][key].copy()
            except:
                return self._zsic_cfg['settings'][key]

