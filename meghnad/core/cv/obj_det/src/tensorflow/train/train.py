import time
import os
import math
import sys

import tensorflow as tf
from utils import ret_values
from utils.log import Log
from meghnad.core.cv.obj_det.src.tensorflow.train.eval import TfObjDetEval
from meghnad.core.cv.obj_det.src.tensorflow.model_loader.ssd.utils.ssd_loss_utils import SSDLoss
from meghnad.core.cv.obj_det.src.tensorflow.data_loader.data_loader import DataLoader
from meghnad.core.cv.obj_det.cfg import ObjDetConfig
from utils.common_defs import class_header, method_header

from .select_model import ObjDetSelectModel


__all__ = ['TfObjDetTrn']


log = Log()


@tf.function
def train_step(imgs, gt_confs, gt_locs, model, criterion, optimizer, weight_decay):
    """Process a training step.

    Parameters
    ----------
    imgs : _type_
        _description_
    gt_confs : _type_
        _description_
    gt_locs : _type_
        _description_
    model : _type_
        _description_
    criterion : _type_
        _description_
    optimizer : _type_
        _description_
    weight_decay : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
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
    """Process a testing step

    Parameters
    ----------
    imgs : _type_
        _description_
    gt_confs : _type_
        _description_
    gt_locs : _type_
        _description_
    model : _type_
        _description_
    criterion : _type_
        _description_
    weight_decay : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    confs, locs = model(imgs, training=False)

    conf_loss, loc_loss = criterion(
        confs, locs, gt_confs, gt_locs)

    loss = conf_loss + loc_loss
    l2_loss = [tf.nn.l2_loss(t) for t in model.trainable_variables]
    l2_loss = weight_decay * tf.math.reduce_sum(l2_loss)
    loss += l2_loss

    return loss, conf_loss, loc_loss, l2_loss


def load_config_from_settings(settings):
    settings = [f'{setting}_models' for setting in settings]
    cfg_obj = ObjDetConfig()
    data_cfg = cfg_obj.get_data_cfg()

    model_cfgs = []
    for setting in settings:
        model_settings = cfg_obj.get_model_settings(setting)
        model_names = model_settings[setting]
        for model_name in model_names:
            model_cfg = cfg_obj.get_model_cfg(model_name)
            model_cfgs.append(model_cfg)
    return model_cfgs


@class_header()
class TfObjDetTrn:
    def __init__(self, settings):
        self.settings = settings
        self.model_cfgs, self.data_cfg = load_config_from_settings(settings)
        self.model_selection = ObjDetSelectModel(self.model_cfgs)
        self.data_loaders = None

    @method_header()
    def config_connectors(self, data_path: str) -> None:
        self.data_loaders = [DataLoader(self.data_cfg, model_cfg)
                             for model_cfg in self.model_cfgs]

    @method_header()
    def train(self, epochs: int = 10) -> object:
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

        # if self.model is None:
        #     log.ERROR(sys._getframe().f_lineno,
        #               __file__, __name__, "Model not initialized")
        #     return ret_values.IXO_RET_INVALID_INPUTS
        # if not os.path.isdir(self.checkpoint_dir):
        #     os.makedirs(self.checkpoint_dir, exist_ok=True)

        # model_name = self.model_loader.aarch

        # # Found a checkpoint
        # ckpt = tf.train.Checkpoint(
        #     model=self.model, optimizer=self.optimizer, start_epoch=tf.Variable(0))
        # if self.resume_path:
        #     ckpt.read(self.resume_path)
        #     print(f'Resume training from {self.resume_path}')
        # else:
        #     if self.checkpoint_dir:
        #         ckpt_filenames = os.listdir(self.checkpoint_dir)
        #         prefix = f'{model_name}_last.ckpt'
        #         for filename in ckpt_filenames:
        #             if filename.startswith(prefix):
        #                 ckpt_path = os.path.join(self.checkpoint_dir, prefix)
        #                 ckpt.read(ckpt_path)
        #                 print(f'Resume training from {ckpt_path}')
        #                 break

        for data_loader, (model, optimizer, loss) in zip(data_loader, self.model_selection.models):
            # train_log_dir = os.path.join(self.log_dir, 'train')
            # val_log_dir = os.path.join(self.log_dir, 'val')
            # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            # val_summary_writer = tf.summary.create_file_writer(val_log_dir)

            base_lr = float(optimizer.learning_rate.numpy())
            warmup_learning_rate = base_lr / 6
            warmup_steps = 2000
            optimizer.learning_rate.assign(warmup_learning_rate)
            steps_per_epoch = max(
                1, self.data_loader.train_size // self.data_loader.batch_size)
            total_steps = epochs * steps_per_epoch
            global_step = 0

            print('Steps per epoch', steps_per_epoch)
            print('Total steps', total_steps)

            # Start training
            best_map = 0
            start_epoch = ckpt.start_epoch.numpy()
            for epoch in range(start_epoch, epochs):
                avg_loss = 0.0
                avg_conf_loss = 0.0
                avg_loc_loss = 0.0
                avg_l2_loss = 0.0
                start = time.time()
                for i, (imgs, gt_confs, gt_locs) in enumerate(data_loader.train_dataset):
                    # Forward + Backward
                    loss, conf_loss, loc_loss, l2_loss = train_step(
                        imgs, gt_confs, gt_locs,
                        model, self.loss, self.optimizer, self.weight_decay
                    )

                    # Compute average losses
                    avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
                    avg_conf_loss = (avg_conf_loss * i +
                                     conf_loss.numpy()) / (i + 1)
                    avg_loc_loss = (avg_loc_loss * i +
                                    loc_loss.numpy()) / (i + 1)
                    avg_l2_loss = (avg_l2_loss * i + l2_loss.numpy()) / (i + 1)
                    if (i + 1) % self.print_every == 0:
                        print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f} L2 Loss {:.4f}'.format(
                            epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss, avg_l2_loss))

                    # Learning rate scheduler
                    global_step = epoch * steps_per_epoch + i + 1
                    if global_step <= warmup_steps:
                        slope = (base_lr - warmup_learning_rate) / warmup_steps
                        new_lr = warmup_learning_rate + slope * \
                            tf.cast(global_step, tf.float32)
                        self.optimizer.learning_rate.assign(new_lr)
                    else:
                        new_lr = 0.5 * base_lr * (1 + tf.cos(
                            math.pi *
                            (tf.cast(i + 1, tf.float32) - warmup_steps
                             ) / float(total_steps - warmup_steps)))
                        self.optimizer.learning_rate.assign(new_lr)
                print('Current learning rate:',
                      self.optimizer.learning_rate.numpy())

                # Start evaluation at the end of epoch
                print('Evaluating...')
                map, map50 = self.evaluator.eval()

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', avg_loss, step=epoch)
                    tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
                    tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

                with val_summary_writer.as_default():
                    tf.summary.scalar('mAP', map, step=epoch)

                # Checkpoint
                ckpt.start_epoch.assign_add(1)
                save_path = ckpt.write(os.path.join(
                    self.checkpoint_dir, f'{model_name}_last.ckpt'))
                print("Saved checkpoint for epoch {}: {}".format(
                    int(ckpt.start_epoch), save_path))

                # Save the best
                if map > best_map:
                    best_map = map
                    self.best_path = ckpt.write(os.path.join(
                        self.checkpoint_dir, f'{model_name}_best.ckpt'))
                    print(f'Saved best model as {self.best_path}')

                self.total_epochs_ran += 1
            return ret_values.IXO_RET_SUCCESS

    def get_best_model(self):
        return self.best_path
