#######################################################################################################################
# Configurations for Object Detection.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Prudhvi Raju
#######################################################################################################################
from typing import List
from meghnad.cfg.config import MeghnadConfig

__all__ = ['ObjDetConfig']

_obj_det_cfg = {
    'models':
    {
        'ssd_mobilenetv2_300',
        'ssd_mobilenetv2_512',
        'EfficientNetB3',
        'EfficientNetB4',
        'EfficientNetB5',
        'EfficientNetV2S',
        'EfficientNetV2M',
        'EfficientNetV2L',
    },
    'model_cfg':
    {
        'ssd_mobilenetv2_300':
        {
            'arch': 'ssd',
            'backbone': 'mobilenetv2',
            'loss': 'SSDLoss',
            'pretrained': None,
            'img_size': 300,
            'include_background': True,
            'feature_map_shapes': [19, 10, 5, 3, 2, 1],
            'aspect_ratios': [[1., 2., 1. / 2.],
                              [1., 2., 1. / 2., 3., 1. / 3.],
                              [1., 2., 1. / 2., 3., 1. / 3.],
                              [1., 2., 1. / 2., 3., 1. / 3.],
                              [1., 2., 1. / 2.],
                              [1., 2., 1. / 2.]],
            'iou_threshold': 0.5,
            'score_threshold': 0.4,
            'neg_pos_ratio': 3,
            'loc_loss_alpha': 1,
            'variances': [0.1, 0.1, 0.2, 0.2],
            'augmentations': {
                'train': {
                    'random_brightness': {},
                    'random_contrast': {},
                    'random_hue': {},
                    'random_saturation': {},
                    'patch': {},
                    'flip_horizontally': {}
                }
            },
            'hyp_params':
            {
                'batch_size': 8,
                'optimizer':
                {
                    'name': 'Adam',
                    'learning_rate': 0.001,
                },
                'weight_decay': 5e-4
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
                'optimizer':
                {
                    'name': 'Adam',
                    'learning_rate': 0.001,
                },
                'weight_decay': 5e-4
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
                'weight_decay': 5e-4
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
                'weight_decay': 5e-4
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
                'weight_decay': 5e-4
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
                'weight_decay': 5e-4
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
                'weight_decay': 5e-4
            }
        }
    },
    'model_settings':
    {
        'default_models': ['ssd_mobilenetv2_300'],
        'light_models': ['ssd_mobilenetv2_300']
    }
}


class ObjDetConfig(MeghnadConfig):
    def __init__(self):
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
        if setting_name and setting_name + '_models' in _obj_det_cfg['model_settings']:
            try:
                return _obj_det_cfg['model_settings'][setting_name].copy()
            except:
                return _obj_det_cfg['model_settings']['default_models']

    def set_user_cfg(self, user_cfg):
        for key in user_cfg:
            self.user_cfg[key] = user_cfg[key]

    def get_user_cfg(self) -> dict:
        return self.user_cfg
