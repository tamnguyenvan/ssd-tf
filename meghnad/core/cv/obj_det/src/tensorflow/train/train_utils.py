import sys
from copy import copy
from typing import List, Dict

import tensorflow as tf

from utils import ret_values
from utils.log import Log
from utils.common_defs import method_header
from meghnad.core.cv.obj_det.cfg import ObjDetConfig

log = Log()


@method_header(
    description='''
        Function to get optimizer''',
    arguments='''
        name: select optimizer by default (adam) is selected
        ''',
    returns='''
        tensorflow optimizer
        ''')
def build_optimizer(cfg: Dict, **kwargs):
    # Update hyper-parameters from kwargs
    opt_name = cfg['hyp_params']['optimizer']['name']
    if 'optimizer' in kwargs:
        opt_name = kwargs['optimizer']

    opt_params = copy(cfg['hyp_params']['optimizer'])
    del opt_params['name']
    if 'optimizer_params' in kwargs:
        opt_params = kwargs['optimizer_params']

    if opt_name == 'Adam':
        return tf.keras.optimizers.Adam(
            learning_rate=opt_params.get('learning_rate', 1e-3),
            beta_1=opt_params.get('beta_1', 0.9),
            beta_2=opt_params.get('beta_2', 0.999),
            epsilon=opt_params.get('epsilon', 1e-7)
        )
    elif opt_name == 'SGD':
        return tf.keras.optimizers.SGD(**opt_params)
    else:
        log.ERROR(sys._getframe().f_lineno,
                  __file__, __name__, f"Unsupported optimizer {opt_name}")
        return ret_values.IXO_RET_NOT_SUPPORTED


def build_scheduler():
    def schedule(epoch):
        """Generating learning rate value for a given epoch.
        inputs:
            epoch = number of current epoch

        outputs:
            learning_rate = float learning rate value
        """
        if epoch < 100:
            return 1e-3
        elif epoch < 125:
            return 1e-4
        else:
            return 1e-5
    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)


def get_model_cfgs_from_settings(settings: List[str]) -> List[Dict]:
    cfg = ObjDetConfig()
    model_cfgs = []
    for setting in settings:
        model_names = cfg.get_model_settings(setting)
        for model_name in model_names:
            model_cfg = cfg.get_model_cfg(model_name)
            model_cfgs.append(model_cfg)
    return model_cfgs
