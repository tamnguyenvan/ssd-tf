import os
import sys

import tensorflow as tf

from utils import ret_values
from utils.log import Log
import meghnad.core.cv.obj_det.cfg.config as cfg

from .models.retinanet import RetinaNet


log = Log()


class ModelLoader:
    def __init__(self,
                 aarch,
                 num_classes,
                 saved_weights_path=None,
                 initialize_weight=False,
                 input_shape=(320, 320, 3),
                 trainable=False,
                 pooling_type=None,
                 saved_model_path=None):
        self.aarch = aarch
        self.model = None
        self.saved_weights_path = saved_weights_path
        self.initialize_weight = initialize_weight
        self.input_shape = input_shape
        self.trainable = trainable
        self.pooling_type = pooling_type
        self.num_classes = num_classes
        self.saved_model_path = saved_model_path

    def load_model(self):
        weights = None if self.initialize_weight else self.saved_weights_path if self.saved_weights_path is not None else 'imagenet'
        try:
            models = cfg.ObjDetConfig().get_models_by_names()
            print(self.aarch)
            print(self.aarch not in models)
            print("path", tf.__path__)
            print(tf.__version__)
            if self.aarch not in models:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__, "Invalid model selected")
                return ret_values.IXO_RET_INVALID_INPUTS

            model = RetinaNet(self.num_classes, self.aarch)
            # func = globals()[self.aarch]
            # print("function", func)
            # print(tf.__version__)

            # model = tf.keras.Sequential()
            # if self.trainable:
            #     model.add(func(
            #         input_shape=self.input_shape,
            #         weights=weights,
            #         include_top=False,
            #         classes=self.num_classes
            #     ))
            # else:
            #     model.add(func())
            # model.add(tf.keras.layers.Flatten())
            # model.add(tf.keras.layers.Dense(4))
        except Exception as e:
            print(e)
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, e)
            return ret_values.IXO_RET_INVALID_INPUTS

        self.model = model
        # self.model.build((None,) + self.input_shape)
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
