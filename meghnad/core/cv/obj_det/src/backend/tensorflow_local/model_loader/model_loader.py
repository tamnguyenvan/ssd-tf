import os
import sys

import tensorflow as tf

from utils import ret_values
from utils.log import Log
import meghnad.core.cv.obj_det.cfg.config as cfg

from .models import ssd


log = Log()


class ModelLoader:
    def __init__(self,
                 aarch,
                 num_classes,
                 model_config,
                 weights=None):
        self.aarch = aarch
        self.model = None
        self.num_classes = num_classes
        self.model_config = model_config
        self.weights = weights

    def load_model(self):
        try:
            config = cfg.ObjDetConfig()
            models = config.get_models_by_names()
            if self.aarch not in models:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__, "Invalid model selected")
                return ret_values.IXO_RET_INVALID_INPUTS

            model = ssd(self.aarch,
                        self.model_config['input_shape'],
                        self.num_classes,
                        self.model_config['num_anchors'])
            if self.weights:
                model.load_weights(self.weights)
        except Exception as e:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, e)
            return ret_values.IXO_RET_INVALID_INPUTS

        self.model = model
        return ret_values.IXO_RET_SUCCESS, self.model

    def load_model_from_url(self, url, model_dir=None):
        path_to_downloaded_dataset = tf.keras.utils.get_file(
            origin=url, extract=True, cache_dir=model_dir)
        path = os.path.dirname(path_to_downloaded_dataset)
        if not os.path.exists(path):
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Invalid path for data: {} ".format(path))
            return ret_values.IXO_RET_INVALID_INPUTS
        self.saved_model_path = path
        return ret_values.IXO_RET_SUCCESS, self.saved_model_path, self.model

    def load_model_from_directory(self, filepath, compile):
        if not os.path.exists(filepath):
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Invalid path for model: {} ".format(filepath))
            return ret_values.IXO_RET_INVALID_INPUTS
        model = tf.keras.models.load_model(filepath, compile=compile)
        self.model = model

    def save_model_to_directory(self, model, file_path, overwrite):
        if not self.model:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "model not initialized")
            return ret_values.IXO_RET_INVALID_INPUTS
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        model.save(file_path,
                   overwrite=overwrite,
                   include_optimizer=True)
