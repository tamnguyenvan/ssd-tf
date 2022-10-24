#######################################################################################################################
# Default configurations for Meghnad.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.common_defs import *
from utils import ret_values
from utils.log import Log

import sys
import os
from pathlib import Path

log = Log()

_meghnad_cfg =\
    {
        'DEFAULT_LOG_LEVELS':
        {
            'TF_CPP_MIN_LOG_LEVEL': '3',
        },
        'DEFAULT_EXT_BIN_PATHS':
        {
            'EXT_PATH': 'bin/ext/',
            'NLTK_PATH': 'nltk/',
            'FLAIR_PATH': 'flair/',
            'FASTPUNCT_PATH': 'fastpunct/',
            'PYANNOTE_PATH': 'pyannote/',
            'TF_HUB_PATH': 'tf/hub/',
            'HF_HOME_PATH': 'hf/',
            'HF_TRANSFORMERS_PATH': 'hf/transformers/',
            'HF_HUB_PATH': 'hf/hub/',
        },
        'DEFAULT_INT_BIN_PATHS':
        {
            'INT_PATH': 'bin/int/',
        },
        'DEFAULT_REPO_PATHS':
        {
            'REPO_PATH': 'repo/',
            'OBJ_DET_REPO_PATH': 'obj_det/',
        },
        'DEFAULT_MISC':
        {
            'VAL_SIZE': 0.25,
            'MAX_GPU_ALLOC_PCT': 0.5,
        },
    }


class MeghnadConfig():
    def __init__(self, *args, **kwargs):
        self.base_dir = str(Path(os.path.dirname(
            __file__) + "/").parent.absolute()) + "/"
        self.ext_path = self.base_dir + \
            _meghnad_cfg['DEFAULT_EXT_BIN_PATHS']['EXT_PATH']
        self.int_path = self.base_dir + \
            _meghnad_cfg['DEFAULT_INT_BIN_PATHS']['INT_PATH']
        self.repo_path = self.base_dir + \
            _meghnad_cfg['DEFAULT_REPO_PATHS']['REPO_PATH']

    def set_gpu(self, dl_framework: str = 'tf', gpu_alloc_in_mb: int = None) -> (int, int):
        ret_val = ret_values.IXO_RET_SUCCESS
        gpu_id = None

        if dl_framework == 'tf' or dl_framework == 'pt':
            self.dl_framework = dl_framework
        else:
            self.dl_framework = 'tf'
            log.WARNING(sys._getframe().f_lineno,
                        __file__, __name__,
                        "DL framework {} not supported. Defaulting to tf".format(dl_framework))

        if not gpu_alloc_in_mb or gpu_alloc_in_mb < 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            import tensorflow as tf
            import torch

            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    torch.cuda.empty_cache()
                    gpu_id = 0

                    for gpu in gpus:
                        total_memory = torch.cuda.get_device_properties(
                            gpu_id).total_memory / 1e6
                        gpu_alloc_in_pct = gpu_alloc_in_mb / total_memory

                        if gpu_alloc_in_pct <= self.get_meghnad_configs('MAX_GPU_ALLOC_PCT'):
                            if self.dl_framework == 'tf':
                                gpu_config = tf.config.experimental.VirtualDeviceConfiguration(
                                    memory_limit=gpu_alloc_in_mb)
                                tf.config.experimental.set_virtual_device_configuration(gpu, [
                                                                                        gpu_config])
                            else:
                                torch.cuda.set_per_process_memory_fraction(
                                    gpu_alloc_in_pct, gpu_id)

                            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                            break
                        else:
                            if self.dl_framework == 'tf':
                                tf.config.experimental.set_memory_growth(
                                    gpu, True)
                            else:
                                log.ERROR(sys._getframe().f_lineno,
                                          __file__, __name__,
                                          "Not implemented")
                                ret_val = ret_values.IXO_RET_INCORRECT_CONFIG

                        gpu_id += 1
                except RuntimeError as e:
                    log.ERROR(sys._getframe().f_lineno,
                              __file__, __name__,
                              "{}".format(e))
                    ret_val = ret_values.IXO_RET_GENERIC_FAILURE
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                ret_val = ret_values.IXO_RET_GENERIC_FAILURE

        if ret_val != ret_values.IXO_RET_SUCCESS:
            gpu_id = None

        return ret_val, gpu_id

    def set_base_environ(self):
        # Hide GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # Set up external model paths through environment variable
        os.environ["TFHUB_CACHE_DIR"] = self.get_meghnad_configs('TF_HUB_PATH')

        os.environ["HF_HOME"] = self.get_meghnad_configs('HF_HOME_PATH')
        os.environ["TRANSFORMERS_CACHE"] = self.get_meghnad_configs(
            'HF_TRANSFORMERS_PATH')
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.get_meghnad_configs(
            'HF_TRANSFORMERS_PATH')

        os.environ["FLAIR_CACHE_ROOT"] = self.get_meghnad_configs('FLAIR_PATH')

        # # Suppress unnecessary warnings
        # if ret_values.IXO_MEGHNAD_LOG_LEVEL < ret_values.IXO_LOG_VERBOSE:
        #     os.environ["TF_CPP_MIN_LOG_LEVEL"] = self.get_meghnad_configs(
        #         'TF_CPP_MIN_LOG_LEVEL')

        #     import fasttext
        #     fasttext.FastText.eprint = lambda x: None

        #     from transformers import logging
        #     logging.set_verbosity_error()

        #     import warnings
        #     warnings.filterwarnings('ignore')

        # Set random seeds
        _set_seed()

        # External model paths
        import nltk
        nltk.data.path.append(self.get_meghnad_configs('NLTK_PATH'))

    def get_meghnad_configs(self, key: str) -> object:
        val = None

        if key:
            if key == 'BASE_DIR':
                val = self.base_dir
            elif key == 'EXT_PATH':
                val = self.ext_path
            elif key == 'INT_PATH':
                val = self.int_path
            elif key == 'REPO_PATH':
                val = self.repo_path
            else:
                for section in _meghnad_cfg:
                    if key in _meghnad_cfg[section]:
                        val = _meghnad_cfg[section][key]

                        if key == 'NLTK_PATH':
                            val = self.ext_path + val
                        elif key == 'FLAIR_PATH':
                            val = self.ext_path + val
                        elif key == 'FASTPUNCT_PATH':
                            val = self.ext_path + val
                        elif key == 'PYANNOTE_PATH':
                            val = self.ext_path + val
                        elif key == 'TF_HUB_PATH':
                            val = self.ext_path + val
                        elif key == 'HF_HOME_PATH':
                            val = self.ext_path + val
                        elif key == 'HF_HUB_PATH':
                            val = self.ext_path + val
                        elif key == 'HF_TRANSFORMERS_PATH':
                            val = self.ext_path + val
                        elif key == 'OBJ_DET_REPO_PATH':
                            val = self.repo_path + val

        if not val:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Key {} not found".format(key))

        try:
            return val.copy()
        except:
            return val


def _set_seed():
    import tensorflow as tf
    tf.random.set_seed(ret_values.IXO_SEED)

    import numpy as np
    np.random.seed(ret_values.IXO_SEED)
