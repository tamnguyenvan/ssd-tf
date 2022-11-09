#######################################################################################################################
# Configurations for Object Detection.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Prudhvi Raju
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.cfg.config import MeghnadConfig

_obj_det_cfg = {
    'model_names':
        {
            'MobileNet',
            'EfficientNet',
            'EfficientNetV2',
        },
    'model_archs':
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
            'train_dir': None,
            'test_dir': None,
            'val_dir': None,
            'num_classes': 10 + 1,  # num_classes + background
            'train_test_val_split': (0.7, 0.2, 0.1),
        },
    'model_cfg':
        {
            'MobileNetV2': {
                'input_shape': (300, 300, 3),
                'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                'num_anchors': [4, 6, 6, 6, 4, 4],
                'feature_map_sizes': [19, 10, 5, 3, 2, 1],
                'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.05],
                'neg_ratio': 3,
            },
            'EfficientNetB3': {
                'input_shape': (512, 512, 3),
                'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
                'num_anchors': [4, 6, 6, 4, 4],
                'feature_map_sizes': [16, 8, 4, 2, 1],
                'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
                'neg_ratio': 3,
            },
            'EfficientNetB4': {
                'input_shape': (512, 512, 3),
                'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
                'num_anchors': [4, 6, 6, 4, 4],
                'feature_map_sizes': [16, 8, 4, 2, 1],
                'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
                'neg_ratio': 3,
            },
            'EfficientNetB5': {
                'input_shape': (512, 512, 3),
                'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
                'num_anchors': [4, 6, 6, 4, 4],
                'feature_map_sizes': [16, 8, 4, 2, 1],
                'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
                'neg_ratio': 3,
            },
            'EfficientNetV2S': {
                'input_shape': (512, 512, 3),
                'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
                'num_anchors': [4, 6, 6, 4, 4],
                'feature_map_sizes': [16, 8, 4, 2, 1],
                'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
                'neg_ratio': 3,
            },
            'EfficientNetV2M': {
                'input_shape': (512, 512, 3),
                'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
                'num_anchors': [4, 6, 6, 4, 4],
                'feature_map_sizes': [16, 8, 4, 2, 1],
                'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
                'neg_ratio': 3,
            },
            'EfficientNetV2L': {
                'input_shape': (512, 512, 3),
                'aspect_ratios': [[2], [2, 3], [2, 3], [2], [2]],
                'num_anchors': [4, 6, 6, 4, 4],
                'feature_map_sizes': [16, 8, 4, 2, 1],
                'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
                'neg_ratio': 3,
            }
        },
    'model_params':
        {
            'MobileNetV2': {
                'metrics': ['map'],
                'batch_size': 8,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            },
            'EfficientNetB3': {
                'metrics': ['map'],
                'batch_size': 1,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            },
            'EfficientNetB4': {
                'metrics': ['map'],
                'batch_size': 2,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            },
            'EfficientNetB5': {
                'metrics': ['map'],
                'batch_size': 1,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            },
            'EfficientNetV2S': {
                'metrics': ['map'],
                'batch_size': 2,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            },
            'EfficientNetV2M': {
                'metrics': ['map'],
                'batch_size': 2,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            },
            'EfficientNetV2L': {
                'metrics': ['map'],
                'batch_size': 1,
                'optimizer': 'adam',
                'learning_rate': 0.0001,
            }
        },
}


class ObjDetConfig(MeghnadConfig):
    def __init__(self, *args):
        super().__init__()

        self.user_cfg = {}
        self.user_cfg['store_tensorboard_logs'] = False
        self.user_cfg['log_dir'] = False
        self.user_cfg['prediction_postprocessing'] = False

    def get_model_cfg(self, model_name):
        try:
            return _obj_det_cfg['model_cfg'].copy()[model_name]
        except:
            return _obj_det_cfg['model_cfg'][model_name]

    def get_model_params(self, model_name):
        try:
            return _obj_det_cfg['model_params'].copy()[model_name]
        except:
            return _obj_det_cfg['model_params'][model_name]

    def get_data_cfg(self):
        try:
            return _obj_det_cfg['data_cfg'].copy()
        except:
            return _obj_det_cfg['data_cfg']

    def set_user_cfg(self, user_cfg):
        for key in user_cfg:
            self.user_cfg[key] = user_cfg[key]

    def get_user_cfg(self):
        return self.user_cfg

    def get_models_by_names(self) -> [str]:
        models = []
        for key in _obj_det_cfg['model_archs']:
            models.append(key)
        return models

