import tensorflow as tf
from utils import ret_values
from utils.log import Log
from utils.common_defs import method_header
import sys

log = Log()


__all__ = ['get_optimizer']


@method_header(
    description='''
        Function to get optimizer''',
    arguments='''
        name: select optimizer by default (adam) is selected
        ''',
    returns='''
        tensorflow optimizer
        ''')
def get_optimizer(name='Adam', **kwargs):
    if name == 'Adam':
        return tf.keras.optimizers.Adam(**kwargs)
    else:
        log.ERROR(sys._getframe().f_lineno,
                  __file__, __name__, f"Unsupported optimizer {name}")
        return ret_values.IXO_RET_NOT_SUPPORTED
