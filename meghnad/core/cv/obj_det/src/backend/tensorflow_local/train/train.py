import sys

import tensorflow as tf
from utils import ret_values
from utils.log import Log


log = Log()


class ModelTrainer:
    def __init__(self, train_dataset=None, validation_dataset=None, test_dataset=None,
                 model=None, loss='mae', metrics=["accuracy"], learning_rate=0.0001,
                 optimizer="Adam", store_tensorboard_logs=True, log_dir=None, prediction_postprocessing=None):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.total_epochs_ran = 0
        self.prediction_postprocessing = prediction_postprocessing
        self.tensorboard_callback = None
        if store_tensorboard_logs:
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                                       histogram_freq=1)
        self.history = None

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics,
                           experimental_run_tf_function=False)
        self.model.optimizer.learning_rate = self.learning_rate

        return ret_values.IXO_RET_SUCCESS

    def train(self, epochs):
        self.compile_model()
        try:
            epochs = int(epochs)
            if epochs <= 0:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__, "Epochs value must be a positive integer")
                return ret_values.IXO_RET_INVALID_INPUTS

        except ValueError:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Epochs value must be a positive integer")
            return ret_values.IXO_RET_INVALID_INPUTS

        if self.model is None:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Model not initialized")
            return ret_values.IXO_RET_INVALID_INPUTS
        self.history = self.model.fit(self.train_dataset,
                                      validation_data=self.validation_dataset,
                                      batch_size=1,
                                      epochs=epochs + self.total_epochs_ran,
                                      initial_epoch=self.total_epochs_ran,
                                      verbose=2)

        self.total_epochs_ran += self.history.epoch[-1]
        return ret_values.IXO_RET_SUCCESS
