#######################################################################################################################
# Configurations for speech to text.
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

_zsstt_cfg =\
{
    'models':
    {
        's2t-medium-librispeech-asr':
        {
            'ckpt': 'facebook/s2t-medium-librispeech-asr',
            'size': 285,
            'lang': 'en',
        },
        'wav2vec2-base-960h':
        {
            'ckpt': 'facebook/wav2vec2-base-960h',
            'size': 378,
            'lang': 'en',
        },
        'wav2vec2-large-960h':
        {
            'ckpt': 'facebook/wav2vec2-large-960h',
            'size': 1260,
            'lang': 'en',
        },
        'wav2vec2-large-960h-lv60-self':
        {
            'ckpt': 'facebook/wav2vec2-large-960h-lv60-self',
            'size': 1260,
            'lang': 'en',
        },
        'vakyansh-wav2vec2-hindi-him-4200':
        {
            'ckpt': 'Harveenchadha/vakyansh-wav2vec2-hindi-him-4200',
            'size': 378,
            'lang': 'hi',
        },
        'wav2vec2-large-xlsr-53':
        {
            'ckpt': 'facebook/wav2vec2-large-xlsr-53',
            'size': 1270,
            'lang': 'multi',
        },
    },
    'diarization_models':
    {
        'speaker-segmentation':
        {
            'ckpt': 'pyannote/speaker-segmentation',
            'size': 18,
            'lang': 'multi',
        },
        'speaker-diarization':
        {
            'ckpt': 'pyannote/speaker-diarization',
            'size': 10,
            'lang': 'multi',
        },
    },
    'settings':
    {
        'default_model':
        {
            'en': 'wav2vec2-large-960h-lv60-self',
            'hi': 'vakyansh-wav2vec2-hindi-him-4200',
            'multi': 'wav2vec2-large-xlsr-53',
        },
        'default_diarization_model':
        {
            'high_acc': 'speaker-segmentation',
            'num_speakers_support': 'speaker-diarization',
        },
        'stream_split_slice_len_in_secs': 60,
        'end_of_slice_marker': '\n',
        'sample_rate': 16000,
    },
}

class ZeroShotSTTConfig(MeghnadConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()

        try:
            self.zsstt_models = _zsstt_cfg['models'].copy()
        except:
            self.zsstt_models = _zsstt_cfg['models']

        if 'mode' in kwargs and kwargs['mode'] == 'diarization':
            self.diarization = True
            try:
                self.zsstt_diarization_models = _zsstt_cfg['diarization_models'].copy()
            except:
                self.zsstt_diarization_models = _zsstt_cfg['diarization_models']
        else:
            self.diarization = False

    def get_zsstt_model(self, model_name:str=None, lang:str='en') -> dict:
        if model_name and model_name in self.zsstt_models:
            try:
                return self.zsstt_models[model_name].copy()
            except:
                return self.zsstt_models[model_name]
        else:
            default_models = self.get_zsstt_settings('default_model')
            if lang not in default_models:
                lang = 'multi'
            default_model = default_models[lang]

            try:
                return self.zsstt_models[default_model].copy()
            except:
                return self.zsstt_models[default_model]

    def get_zsstt_diarization_model(self, model_name:str=None, num_speakers_known:bool=False) -> dict:
        if self.diarization:
            if model_name and model_name in self.zsstt_diarization_models:
                try:
                    return self.zsstt_diarization_models[model_name].copy()
                except:
                    return self.zsstt_diarization_models[model_name]
            else:
                if num_speakers_known:
                    default_model = self.get_zsstt_settings('default_diarization_model')['num_speakers_support']
                else:
                    default_model = self.get_zsstt_settings('default_diarization_model')['high_acc']

                try:
                    return self.zsstt_diarization_models[default_model].copy()
                except:
                    return self.zsstt_diarization_models[default_model]
        else:
            return None

    def get_zsstt_settings(self, key:str) -> str:
        if key and key in _zsstt_cfg['settings']:
            try:
                return _zsstt_cfg['settings'][key].copy()
            except:
                return _zsstt_cfg['settings'][key]

