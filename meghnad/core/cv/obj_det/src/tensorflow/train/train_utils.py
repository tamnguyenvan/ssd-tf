import tensorflow as tf
from utils import ret_values
from utils.log import Log
import sys

log = Log()


def get_optimizer(name='Adam', **kwargs):
    if name == 'Adam':
        return tf.keras.optimizers.Adam(**kwargs)
    else:
        log.ERROR(sys._getframe().f_lineno,
                  __file__, __name__, f"Unsupported optimizer {name}")
        return ret_values.IXO_RET_NOT_SUPPORTED
