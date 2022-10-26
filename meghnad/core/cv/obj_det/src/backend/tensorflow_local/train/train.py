import time
import os
import sys

import tensorflow as tf
from utils import ret_values
from utils.log import Log
from .train_utils import get_optimizer


log = Log()


@tf.function
def train_step(imgs, gt_confs, gt_locs, model, criterion, optimizer, weight_decay):
    with tf.GradientTape() as tape:
        confs, locs = model(imgs)

        conf_loss, loc_loss = criterion(
            confs, locs, gt_confs, gt_locs)

        loss = conf_loss + loc_loss
        l2_loss = [tf.nn.l2_loss(t) for t in model.trainable_variables]
        l2_loss = weight_decay * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, conf_loss, loc_loss, l2_loss


@tf.function
def test_step(imgs, gt_confs, gt_locs, model, criterion, weight_decay):
    confs, locs = model(imgs, training=False)

    conf_loss, loc_loss = criterion(
        confs, locs, gt_confs, gt_locs)

    loss = conf_loss + loc_loss
    l2_loss = [tf.nn.l2_loss(t) for t in model.trainable_variables]
    l2_loss = weight_decay * tf.math.reduce_sum(l2_loss)
    loss += l2_loss

    return loss, conf_loss, loc_loss, l2_loss


class ModelTrainer:
    def __init__(self,
                 train_dataset=None,
                 validation_dataset=None,
                 test_dataset=None,
                 model=None,
                 loss='mae',
                 metrics=["accuracy"],
                 learning_rate=0.001,
                 optimizer="Adam",
                 weight_decay=1e-4,
                 store_tensorboard_logs=True,
                 log_dir=None,
                 prediction_postprocessing=None):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = get_optimizer(optimizer, learning_rate=learning_rate)
        self.weight_decay = weight_decay
        self.loss = loss
        self.metrics = metrics
        self.total_epochs_ran = 0
        self.prediction_postprocessing = prediction_postprocessing
        self.tensorboard_callback = None
        self.log_dir = log_dir if log_dir else 'training_logs'
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
        # self.history = self.model.fit(self.train_dataset,
        #                               validation_data=self.validation_dataset,
        #                               batch_size=1,
        #                               epochs=epochs + self.total_epochs_ran,
        #                               initial_epoch=self.total_epochs_ran,
        #                               verbose=2)

        train_log_dir = os.path.join(self.log_dir, 'train')
        val_log_dir = os.path.join(self.log_dir, 'val')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        for epoch in range(epochs):
            avg_loss = 0.0
            avg_conf_loss = 0.0
            avg_loc_loss = 0.0
            avg_l2_loss = 0.0
            start = time.time()
            for i, (imgs, gt_confs, gt_locs) in enumerate(self.train_dataset):
                loss, conf_loss, loc_loss, l2_loss = train_step(
                    imgs, gt_confs, gt_locs,
                    self.model, self.loss, self.optimizer, self.weight_decay
                )
                avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
                avg_conf_loss = (avg_conf_loss * i +
                                 conf_loss.numpy()) / (i + 1)
                avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
                avg_l2_loss = (avg_l2_loss * i + l2_loss.numpy()) / (i + 1)
                if (i + 1) % 1 == 0:
                    print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f} L2 Loss {:.4f}'.format(
                        epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss, avg_l2_loss))

            print(f'End epoch {epoch + 1}. Validating...')
            avg_val_loss = 0.0
            avg_val_conf_loss = 0.0
            avg_val_loc_loss = 0.0
            avg_val_l2_loss = 0.0
            for i, (imgs, gt_confs, gt_locs) in enumerate(self.validation_dataset):
                val_loss, val_conf_loss, val_loc_loss, val_l2_loss = test_step(
                    imgs, gt_confs, gt_locs,
                    self.model, self.loss, self.weight_decay
                )
                avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
                avg_val_conf_loss = (avg_val_conf_loss *
                                     i + val_conf_loss.numpy()) / (i + 1)
                avg_val_loc_loss = (avg_val_loc_loss * i +
                                    val_loc_loss.numpy()) / (i + 1)
                avg_val_l2_loss = (avg_val_l2_loss * i +
                                   val_l2_loss.numpy()) / (i + 1)
            print(
                f'Epoch {epoch + 1} | Val_Loss: {avg_val_loss:.4f} '
                f'Val_Conf: {avg_val_conf_loss:.4f} '
                f'Val_Loc: {avg_val_loc_loss:.4f} Val_L2_Loss: {avg_val_l2_loss:.4f}')

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', avg_loss, step=epoch)
                tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
                tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

            with val_summary_writer.as_default():
                tf.summary.scalar('loss', avg_val_loss, step=epoch)
                tf.summary.scalar('conf_loss', avg_val_conf_loss, step=epoch)
                tf.summary.scalar('loc_loss', avg_val_loc_loss, step=epoch)

            if (epoch + 1) % 5 == 0:
                self.model.save_weights(
                    os.path.join(self.log_dir, 'ssd_epoch_{}.h5'.format(epoch + 1)))

            self.total_epochs_ran += 1
        # self.total_epochs_ran += self.history.epoch[-1]
        return ret_values.IXO_RET_SUCCESS
