#######################################################################################################################
# Configurations for NLP classifier.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################
from typing import List
from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.cfg.config import MeghnadConfig

_obj_det_cfg = {
    'modelNames':
    {
        'MobileNet',
        'EfficientNet',
        'EfficientNetV2'
    },
    'modelArchs':
    {
        'ResNet50',
        'MobileNet',
        'MobileNetV2',
        'EfficientNetB3',
        'EfficientNetB4',
        'EfficientNetB5',
        'EfficientNetV2S',
        'EfficientNetV2M',
        'EfficientNetV2L'
    },
    'data_cfg':
    {
        'img_size': (300, 300),
        'train_dir': None,
        'test_dir': None,
        'val_dir': None,
        'train_test_val_split': (0.7, 0.2, 0.1),
    },
    'model_cfg':
    {
        'model': 'MobileNetV2',
        'saved_weights_path': None,
        'initialize_weight': False,
        'input_shape': (300, 300, 3),
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'num_anchors': [4, 6, 6, 6, 4, 4],
        'feature_map_sizes': [19, 10, 5, 3, 2, 1],
        'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.05],
        'neg_ratio': 3,
        'trainable': False,
        'num_classes': 3,
        'saved_model_path': None,
    },
    'model_params':
    {
        'loss': 'mean_absolute_error',
        'metrics': ['accuracy'],
        'cv': 5,
        'batch_size': 32,
        'optimizer': 'adam',
        'learning_rate': 0.001,
    },
    'user_cfg':
    {
        'store_tensorboard_logs': False,
        'log_dir': None,
        'prediction_postprocessing': None
    }

}


class ObjDetConfig(MeghnadConfig):
    def __init__(self, *args):
        super().__init__()
        self.cfg = _obj_det_cfg

    def get_model_cfg(self):
        return self.cfg['model_cfg']

    def get_model_params(self):
        return self.cfg['model_params']

    def get_data_cfg(self):
        return self.cfg['data_cfg']

    def get_user_cfg(self):
        return self.cfg['user_cfg']

    def get_models_by_names(self) -> List:
        models = []
        names = list(_obj_det_cfg['modelArchs'])
        for name in names:
            models.append(name)
        return models
