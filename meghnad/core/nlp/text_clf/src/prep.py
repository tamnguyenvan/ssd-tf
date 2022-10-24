#######################################################################################################################
# Preprocessing helper for text classification training.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
####################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.ret_values import *
from utils.log import Log
from connectors.interfaces.interface import *

import sys
from ast import literal_eval
from joblib import dump
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

log = Log()

class TextClfPrep():
    def __init__(self, configs:object, connector:dict, *args, **kwargs):
        self.configs = configs
        self.connector = connector

        self.data_trn = None
        if 'data_trn' in kwargs:
            self.data_trn = kwargs['data_trn']

        self.data_val = None
        if 'data_val' in kwargs:
            self.data_val = kwargs['data_val']

    def prep_data_trn(self, multi_label:bool=False, val_size:float=None, class_balance:bool=False) -> int:
        if not self.connector:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Connector not configured with any data")
            return IXO_RET_INCORRECT_CONFIG

        if self.connector['multi_features']:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Not Implemented")
            return IXO_RET_NOT_IMPLEMENTED

        self.val_size = val_size
        self.class_balance = class_balance

        if self.connector['data_org'] == 'single_file':
            ret = self._process_single_file_data_trn(multi_label)
            if ret != IXO_RET_SUCCESS:
                return ret
        elif self.connector['data_org'] == 'multi_dir':
            ret = self._process_multi_dir_data_trn(multi_label)
            if ret != IXO_RET_SUCCESS:
                return ret
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Unsupported data_org: {}".format(self.connector['data_org']))
            return IXO_RET_NOT_SUPPORTED

        return IXO_RET_SUCCESS

    # Processing if a single file contains all the training data
    def _process_single_file_data_trn(self, multi_label:bool=False) -> int:
        if not self.connector['feature_cols']:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Connector not configured with any feature column")
            return IXO_RET_INCORRECT_CONFIG

        if not self.connector['target_cols']:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Connector not configured with any target column")
            return IXO_RET_INCORRECT_CONFIG

        if self.connector['data_type'] == 'csv':
            if 'encoding' in self.connector:
                self.data_trn = pd.read_csv(self.connector['data_path'], encoding=self.connector['encoding'])
            else:
                self.data_trn = pd.read_csv(self.connector['data_path'])

            # Basic cleaning
            log.VERBOSE(sys._getframe().f_lineno,
                        __file__, __name__,
                        "Filtering out rows with null values")
            self.data_trn = self.data_trn.dropna()

            log.VERBOSE(sys._getframe().f_lineno,
                        __file__, __name__,
                        "Filtering out instances with single occurrence of corresponding labels")
            self.data_trn = self.data_trn.groupby(self.connector['target_cols']).filter(lambda x: len(x) > 1)
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Unsupported data_type: {} for data_org: {}".format(self.connector['data_type'],
                                                                          self.connector['data_org']))
            return IXO_RET_NOT_SUPPORTED

        if 'val_data_path' in self.connector and self.connector['val_data_path'] is not None:
            if self.connector['data_type'] == 'csv':
                if 'encoding' in self.connector:
                    self.data_val = pd.read_csv(self.connector['val_data_path'], encoding=self.connector['encoding'])
                else:
                    self.data_val = pd.read_csv(self.connector['val_data_path'])
            else:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__,
                          "Unsupported data_type: {}".format(self.connector['data_type']))
                return IXO_RET_NOT_SUPPORTED

        if not multi_label and self.connector['multi_targets']:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Not Implemented")
            return IXO_RET_NOT_IMPLEMENTED

        if multi_label:
            ret = self._numericalize_multi_label()
        else:
            ret = self._numericalize_single_label()

        if ret == IXO_RET_SUCCESS:
            ret = self._split_single_file_data_for_val(multi_label)

        return ret

    # Numericalize target col if the training task does not involve multi-label
    def _numericalize_single_label(self) -> int:
        self.target_one_hot_encoded = False
        target_col = self.connector['target_cols']
        self.num_classes = len(list(set(self.data_trn[target_col])))

        num_levels = len(list(set(self.data_trn[target_col])))
        if num_levels < 2:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Target {} has no variation".format(target_col))
            return IXO_RET_INVALID_INPUTS
        else:
            self.data_trn[target_col] = self.data_trn[target_col].astype('category')
            self.target_map = dict(enumerate(self.data_trn[target_col].cat.categories))
            self.data_trn[target_col] = self.data_trn[target_col].cat.codes

            if self.data_val is not None:
                self.data_val[target_col] = self.data_val[target_col].astype('category')
                self.data_val[target_col] = self.data_val[target_col].cat.codes

        return IXO_RET_SUCCESS

    # Numericalize target cols if the training task involves multi-label
    def _numericalize_multi_label(self) -> int:
        if not self.connector['multi_targets']:
            self._handle_multi_label_single_target()

        self.target_one_hot_encoded = True
        self.num_classes = len(self.connector['target_cols'])

        self.target_map = dict(enumerate(self.connector['target_cols']))

        log.VERBOSE(sys._getframe().f_lineno,
                    __file__, __name__,
                    "Converting each target to have only 2 levels max")

        for target_col in self.connector['target_cols']:
            num_levels = len(list(set(self.data_trn[target_col])))
            if num_levels > 2:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__,
                          "Multiple targets cannot be possible with any target having more than 2 levels")
                return IXO_RET_INVALID_INPUTS
            elif num_levels < 2:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__,
                          "Target {} has no variation".format(target_col))
                return IXO_RET_INVALID_INPUTS
            else:
                self.data_trn[target_col] = self.data_trn[target_col].astype('category').cat.codes

                if self.data_val is not None:
                    self.data_val[target_col] = self.data_val[target_col].astype('category').cat.codes

        return IXO_RET_SUCCESS

    # Handling if the training task involves multi-label, but there is a single target col
    def _handle_multi_label_single_target(self):
        target_col = self.connector['target_cols']

        # Convert labels from strings to lists
        self.data_trn[target_col] = self.data_trn[target_col].apply(lambda x: literal_eval(str(x)))

        # Convert list type target col (single) into binary valued target cols (multiple)
        self.encoder = MultiLabelBinarizer()
        encoded_target = self.encoder.fit_transform(self.data_trn[target_col])

        self.connector['target_cols'] = self.encoder.classes_
        self.data_trn = self.data_trn.drop(target_col, axis=1).join(pd.DataFrame(encoded_target,
                                                                                 columns=self.encoder.classes_,
                                                                                 index=self.data_trn.index))

        if self.data_val is not None:
            self.data_val[target_col] = self.data_val[target_col].apply(lambda x: literal_eval(str(x)))
            encoded_target = self.encoder.transform(self.data_val[target_col])
            self.data_val = self.data_val.drop(target_col, axis=1).join(pd.DataFrame(encoded_target,
                                                                                     columns=self.encoder.classes_,
                                                                                     index=self.data_val.index))

    # Processing if the training data is contained in a directory structure across multiple files
    def _process_multi_dir_data_trn(self, multi_label:bool=False) -> int:
        if self.connector['feature_cols']:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Not Implemented")
            return IXO_RET_NOT_IMPLEMENTED

        if self.connector['target_cols']:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Not Implemented")
            return IXO_RET_NOT_IMPLEMENTED

        if self.connector['data_type'] == 'txt':
            self.buffer_autotune = tf.data.AUTOTUNE
            class_names = tf.keras.utils.text_dataset_from_directory(self.connector['data_path'],
                                                                     seed=IXO_SEED).class_names
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Unsupported data_type: {}".format(self.connector['data_type']))
            return IXO_RET_NOT_SUPPORTED

        if multi_label or self.connector['multi_targets']:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Not Implemented")
            return IXO_RET_NOT_IMPLEMENTED
        else:
            self.target_one_hot_encoded = False
            self.num_classes = len(class_names)
            self.target_map = dict(enumerate(class_names))

        return IXO_RET_SUCCESS

    def _split_single_file_data_for_val(self, multi_label:bool=False) -> int:
        if 'val_data_path' in self.connector and self.connector['val_data_path'] is not None:
            self.x_trn = self.data_trn[self.connector['feature_cols']]
            self.y_trn = self.data_trn[self.connector['target_cols']]

            self.x_val = self.data_val[self.connector['feature_cols']]
            self.y_val = self.data_val[self.connector['target_cols']]
        else:
            if not self.val_size:
                val_size = self.configs.get_meghnad_configs('VAL_SIZE')
            else:
                val_size = float(self.val_size)

            self.x_trn, self.x_val, self.y_trn, self.y_val =\
                train_test_split(self.data_trn[self.connector['feature_cols']],
                                 self.data_trn[self.connector['target_cols']],
                                 test_size=val_size,
                                 stratify=self.data_trn[self.connector['target_cols']],
                                 random_state=IXO_SEED)

        self.class_weight = None
        if self.class_balance:
            if multi_label:
                log.WARNING(sys._getframe().f_lineno,
                            __file__, __name__,
                            "Not Implemented")
            else:
                self.class_weight = dict(zip(np.unique(self.y_trn),
                                             compute_class_weight(class_weight='balanced',
                                                                  classes=np.unique(self.y_trn),
                                                                  y=self.y_trn)))

        return IXO_RET_SUCCESS

    # Generator for cases where data is present in directory structure
    def gen_multi_dir_data(self, batch_size:int) -> int:
        if 'val_data_path' in self.connector and self.connector['val_data_path'] is not None:
            if self.connector['data_type'] == 'txt':
                self.data_trn = tf.keras.utils.text_dataset_from_directory(self.connector['data_path'],
                                                                           batch_size=batch_size,
                                                                           seed=IXO_SEED)
                self.x_trn = self.data_trn#.cache().prefetch(buffer_size=self.buffer_autotune)
                self.y_trn = None

                self.data_val = tf.keras.utils.text_dataset_from_directory(self.connector['val_data_path'],
                                                                           batch_size=batch_size,
                                                                           seed=IXO_SEED)
                self.x_val = self.data_val#.cache().prefetch(buffer_size=self.buffer_autotune)
                self.y_val = None
            else:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__,
                          "Unsupported data_type: {}".format(self.connector['data_type']))
                return IXO_RET_NOT_SUPPORTED
        else:
            if not self.val_size:
                val_size = self.configs.get_meghnad_configs('VAL_SIZE')
            else:
                val_size = float(self.val_size)

            if self.connector['data_type'] == 'txt':
                self.data_trn = tf.keras.utils.text_dataset_from_directory(self.connector['data_path'],
                                                                           batch_size=batch_size,
                                                                           validation_split=val_size,
                                                                           subset='training',
                                                                           seed=IXO_SEED)
                self.x_trn = self.data_trn#.cache().prefetch(buffer_size=self.buffer_autotune)
                self.y_trn = None

                self.data_val = tf.keras.utils.text_dataset_from_directory(self.connector['data_path'],
                                                                           batch_size=batch_size,
                                                                           validation_split=val_size,
                                                                           subset='validation',
                                                                           seed=IXO_SEED)
                self.x_val = self.data_val#.cache().prefetch(buffer_size=self.buffer_autotune)
                self.y_val = None
            else:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__,
                          "Unsupported data_type: {}".format(self.connector['data_type']))
                return IXO_RET_NOT_SUPPORTED

        self.class_weight = None
        if self.class_balance:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Not Implemented")

        return IXO_RET_SUCCESS

    # Save preprocessing data details to be used during prediction
    def save_prep_data_details(self) -> dict:
        saved_prep_paths = {}

        if self.connector['multi_targets']:
            saved_prep_paths['target_cols'] = self.connector['dir_to_save_model'] + 'target_cols.joblib'
            dump(self.connector['target_cols'], saved_prep_paths['target_cols'])

        saved_prep_paths['target_map'] = self.connector['dir_to_save_model'] + 'target_map.joblib'
        dump(self.target_map, saved_prep_paths['target_map'])

        log.STATUS(sys._getframe().f_lineno,
                   __file__, __name__,
                   "Preprocessing data details saved: {}".format(saved_prep_paths))

        return saved_prep_paths

    # Load preprocessing data details saved during training
    def load_prep_data_details(self):
        pass

if __name__ == '__main__':
    pass

