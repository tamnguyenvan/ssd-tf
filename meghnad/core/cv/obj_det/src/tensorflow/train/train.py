import os
import sys
from typing import List

from tensorflow.keras.callbacks import TensorBoard
from meghnad.core.cv.obj_det.src.tensorflow.data_loader.data_loader import build_dataloader
from meghnad.core.cv.obj_det.src.tensorflow.train.train_utils import build_optimizer, build_scheduler, get_model_cfgs_from_settings
from meghnad.core.cv.obj_det.src.tensorflow.model_loader import build_model, build_losses
from meghnad.core.cv.obj_det.src.tensorflow.train.eval import EvaluationCallback

from utils import ret_values
from utils.log import Log
from utils.common_defs import class_header, method_header


__all__ = ['TFObjDetTrn']


log = Log()


@class_header(
    description='''
        Class for object detection model training''')
class TFObjDetTrn:
    def __init__(self, settings: List[str]) -> None:
        self.settings = settings

    @method_header(
        description='''
                Helper for configuring data connectors.''',
        arguments='''
                data_path: location of the training data (should point to the file in case of a single file, should point to
                the directory in case data exists in multiple files in a directory structure)
                ''')
    def config_connectors(self,
                          data_path: str,
                          dataset: str) -> None:
        self.model_cfgs = get_model_cfgs_from_settings(self.settings)
        self.data_loaders = [build_dataloader(data_path, dataset, model_cfg)
                             for model_cfg in self.model_cfgs]

    @method_header(
        description='''
                Function to set training configurations and start training.''',
        arguments='''
                epochs: set epochs for the training by default it is 10
                checkpoint_dir: directory from where the checkpoints should be loaded
                logdir: directory where the logs should be saved
                resume_path: The path/checkpoint from where the training should be resumed
                print_every: an argument to specify when the function should print or after how many epochs
                ''')
    def train(self,
              batch_size: int = 8,
              epochs: int = 10,
              checkpoint_dir: str = './checkpoints',
              logdir: str = './logs',
              **hyp) -> int:
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

        for model_cfg, data_loader in zip(self.model_cfgs, self.data_loaders):
            # Load datasets
            augmentations = hyp.get('augmentations')
            train_ds, val_ds, test_ds = data_loader.load(
                batch_size, augmentations)

            # Build model from specific config
            model_cfg['num_classes'] = data_loader.num_classes
            model = build_model(model_cfg)

            # Build optimizer
            optimizer = build_optimizer(model_cfg, **hyp)
            losses = build_losses(model_cfg)
            model.compile(
                optimizer=optimizer,
                loss=losses)

            # # Set up model checkpoint
            # checkpoint_dir = f"{checkpoint_dir}/{model_cfg['arch']}_{model_cfg['backbone']}_{model_cfg['img_size']}"
            # os.makedirs(checkpoint_dir, exist_ok=True)
            # checkpoint_callback = ModelCheckpoint(
            #     checkpoint_dir, monitor="val_loss", save_best_only=True, save_weights_only=True)

            # Set up tensorboard
            logdir = f"{logdir}/{model_cfg['arch']}_{model_cfg['backbone']}_{model_cfg['img_size']}"
            tensorboard_callback = TensorBoard(log_dir=logdir)
            learning_rate_callback = build_scheduler()

            # Evaluation
            eval_callbacks = EvaluationCallback(val_ds, data_loader.labels)

            # Train
            model.fit(
                train_ds,
                steps_per_epoch=1,
                epochs=epochs,
                callbacks=[
                    # checkpoint_callback,
                    tensorboard_callback,
                    learning_rate_callback,
                    eval_callbacks
                ]
            )
