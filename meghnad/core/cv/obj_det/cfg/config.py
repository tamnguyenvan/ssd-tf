#######################################################################################################################
# Configurations for Object Detection.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Prudhvi Raju
#######################################################################################################################
from typing import List

from utils.common_defs import *
from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig

__all__ = ['ObjDetConfig']

_obj_det_cfg = {
    'models':
    {
        'MobileNetV2',
        'EfficientNetB3',
        'EfficientNetB4',
        'EfficientNetB5',
        'EfficientNetV2S',
        'EfficientNetV2M',
        'EfficientNetV2L',
    },
    'data_cfg':
    {
        'path': '',
        'train_test_val_split': (0.7, 0.2, 0.1),
    },
    'model_cfg':
    {
        'MobileNetV2':
        {
            'arch': 'MobileNetV2',
            'pretrained': None,
            'input_shape': (300, 300, 3),
            'num_classes': 10 + 1,  # num_classes + background
            'classes': [],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 6, 4, 4],
            'feature_map_sizes': [19, 10, 5, 3, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.05],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'Adam',
                'learning_rate': 0.0001,
            }
        },
        'EfficientNetB3':
        {
            'arch': 'EfficientNetB3',
            'input_shape': (512, 512, 3),
            'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 4, 4],
            'feature_map_sizes': [16, 8, 4, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            }
        },
        'EfficientNetB4': {
            'arch': 'EfficientNetB4',
            'input_shape': (512, 512, 3),
            'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 4, 4],
            'feature_map_sizes': [16, 8, 4, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            }
        },
        'EfficientNetB5': {
            'arch': 'EfficientNetB5',
            'input_shape': (512, 512, 3),
            'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 4, 4],
            'feature_map_sizes': [16, 8, 4, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            }
        },
        'EfficientNetV2S': {
            'arch': 'EfficientNetV2S',
            'input_shape': (512, 512, 3),
            'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 4, 4],
            'feature_map_sizes': [16, 8, 4, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            }
        },
        'EfficientNetV2M': {
            'arch': 'EfficientNetV2M',
            'input_shape': (512, 512, 3),
            'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 4, 4],
            'feature_map_sizes': [16, 8, 4, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            }
        },
        'EfficientNetV2L': {
            'arch': 'EfficientNetV2L',
            'input_shape': (512, 512, 3),
            'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
            'num_anchors': [4, 6, 6, 4, 4],
            'feature_map_sizes': [16, 8, 4, 2, 1],
            'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
            'neg_ratio': 3,
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            }
        }
    },
    'model_settings':
    {
        'default_models': ['MobileNetV2', 'EfficientNetB3'],
        'light_models': ['MobileNetV2']
    }
}


class ObjDetConfig(MeghnadConfig):
    def __init__(self, *args):
        super().__init__()

    def get_model_cfg(self, model_name: str) -> dict:
        try:
            return _obj_det_cfg['model_cfg'].copy()[model_name]
        except:
            return _obj_det_cfg['model_cfg'][model_name]

    def get_data_cfg(self) -> dict:
        try:
            return _obj_det_cfg['data_cfg'].copy()
        except:
            return _obj_det_cfg['data_cfg']

    def get_model_settings(self, setting_name: str = None) -> dict:
        if setting_name and setting_name in _obj_det_cfg['model_settings']:
            try:
                return _obj_det_cfg['model_settings'][setting_name].copy()
            except:
                return _obj_det_cfg['model_settings'][setting_name]

    def set_user_cfg(self, user_cfg):
        for key in user_cfg:
            self.user_cfg[key] = user_cfg[key]

    def get_user_cfg(self) -> dict:
        return self.user_cfg

    def get_models_by_names(self) -> List[str]:
        models = []
        for key in _obj_det_cfg['model_archs']:
            models.append(key)
        return models
