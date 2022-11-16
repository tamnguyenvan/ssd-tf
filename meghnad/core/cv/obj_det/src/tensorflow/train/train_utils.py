import tensorflow as tf


class NotSupportedOptimizer(Exception):
    pass


def get_optimizer(name='Adam', **kwargs):
    if name == 'Adam':
        return tf.keras.optimizers.Adam(**kwargs)
    else:
        NotSupportedOptimizer(f'Not supported optimizer: {name}')
