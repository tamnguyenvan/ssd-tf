#######################################################################################################################
# Training for text classifier.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
####################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from connectors.interfaces.interface import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.text_clf.cfg.config import TextClfConfig
from meghnad.core.nlp.text_clf.src.arch import TextClfArch
from meghnad.core.nlp.text_clf.src.prep import TextClfPrep
from meghnad.core.nlp.text_clf.src.select_model import TextClfSelectModel

import sys, os, gc, shutil, itertools
from joblib import dump
import numpy as np

from sklearn.model_selection import GridSearchCV

log = Log()

@class_header(
description='''
Text classifier training pipeline.''')
class TextClfTrn():
    def __init__(self, multi_label:bool=False, *args, **kwargs):
        self.configs = TextClfConfig(MeghnadConfig())

        self.tf_hub_path = self.configs.get_meghnad_configs('TF_HUB_PATH')
        self.hf_hub_path = self.configs.get_meghnad_configs('HF_HUB_PATH')

        self.multi_label = multi_label

        if 'settings' in kwargs:
            list_settings = kwargs['settings']
        else:
            list_settings = []
        self.model_selection = TextClfSelectModel(self.configs, list_settings)

    @method_header(
        description='''
        Helper for configuring data connectors.''',
        arguments='''
        data_path: location of the training data (should point to the file in case of a single file, should point to 
        the directory in case data exists in multiple files in a directory structure) 
        data_type: type of the training data ('csv' / 'json' / 'txt'); currently 'csv' and 'txt' are supported
        dat_org [optional]: organization of the data ('single_file' / 'multi_dir')
        feature_cols [optional]: attribute names in the data to be used as features during training; currently only
        a single feature column is supported; it is optional in cases such as when the data is present in text files
        under class-specific directories
        target_cols [optional]: attribute names in the data to be used as targets during training (multiple target
        columns supported only in cases where target data is present across multiple columns in one-hot encoded format,
        which is possible if it is a multi label classification task or single label classification task with
        target data present across multiple columns in one-hot encoded format); it is optional in cases such as 
        when the data is present in text files under class-specific directories
        val_data_path [optional]: location of validation data (if separate validation data is provided by the user)
        dir_to_save_model [optional]: location where trained model should be saved''')
    def config_connectors(self, data_path:str, data_type:str,
                          data_org:str='multi_dir',
                          feature_cols:[str]=[], target_cols:[str]=[],
                          val_data_path:str=None,
                          dir_to_save_model:str=None,
                          *args, **kwargs):
        self.connector_trn = {}
        self.connector_trn['data_path'] = data_path
        self.connector_trn['data_type'] = data_type
        if 'encoding' in kwargs:
            self.connector_trn['encoding'] = kwargs['encoding']
        self.connector_trn['data_org'] = data_org

        self.connector_trn['feature_cols'] = feature_cols
        self.connector_trn['multi_features'] = False
        if self.connector_trn['feature_cols']:
            if isinstance(self.connector_trn['feature_cols'], list):
                if len(self.connector_trn['feature_cols']) > 1:
                    self.connector_trn['multi_features'] = True
                else:
                    self.connector_trn['feature_cols'] = self.connector_trn['feature_cols'][0]

        self.connector_trn['target_cols'] = target_cols
        self.connector_trn['multi_targets'] = False
        if self.connector_trn['target_cols']:
            if isinstance(self.connector_trn['target_cols'], list):
                if len(self.connector_trn['target_cols']) > 1:
                    self.connector_trn['multi_targets'] = True
                else:
                    self.connector_trn['target_cols'] = self.connector_trn['target_cols'][0]

        self.connector_trn['val_data_path'] = val_data_path

        if dir_to_save_model:
            self.connector_trn['dir_to_save_model'] = dir_to_save_model
        else:
            self.connector_trn['dir_to_save_model'] = self.configs.get_meghnad_configs('INT_PATH') + 'text_clf/'
            if os.path.exists(self.connector_trn['dir_to_save_model']):
                shutil.rmtree(self.connector_trn['dir_to_save_model'])
            os.mkdir(self.connector_trn['dir_to_save_model'])

        self.prep = TextClfPrep(self.configs, self.connector_trn)

    @method_header(
        description='''
            Training pipeline.''',
        arguments='''
            epochs [optional]: number of epochs
            val_size [optional]: percentage of training data to be used for validation (valid only if no val_data_path 
            parameter was passed to config_connectors)
            class_balance [optional]: whether class weights need to be balanced''',
        returns='''
            a 2 member tuple containing return value and the path where all details related to the trained model
            to be used at the time of prediction got saved''')
    def trn(self, epochs:int=20, val_size:float=None, class_balance:bool=True,
            *args, **kwargs) -> (int, str):
        verbose = IXO_MEGHNAD_LOG_LEVEL >= IXO_LOG_VERBOSE
        self.models = []

        # Params
        params = _get_params(kwargs)

        # Hyper-params permutations
        hyper_params_permute, hyper_params_search_space = _permute_hyper_params(kwargs)

        # Data preparation
        ret = self.prep.prep_data_trn(self.multi_label, val_size, class_balance)
        if ret != IXO_RET_SUCCESS:
            return ret, None

        # Train
        max_val_acc = 0
        final_model = {}

        self.arch = TextClfArch(self.configs, self.prep.num_classes)

        # Top-tune pretrained models
        if self.configs.get_user_cfg()['mode'] != 'emb_fit' or self.connector_trn['data_org'] == 'multi_dir':
            best_model = {}
            for pretrained_model in self.model_selection.pretrained_models:
                top_tuned_model, best_hyper_params = _tune_hyper_params(pretrained_model,
                                                                        params, hyper_params_permute,
                                                                        self.arch, self.prep,
                                                                        self.multi_label, epochs, verbose)

                if top_tuned_model:
                    self.models.append(top_tuned_model)

                    val_acc = max(top_tuned_model['history'].history['val_acc'])
                    if max_val_acc < val_acc:
                        max_val_acc = val_acc

                        best_model['top_tuned_model'] = top_tuned_model.copy()
                        best_model['pretrained_model'] = pretrained_model.copy()
                        best_model['best_hyper_params'] = best_hyper_params.copy()

        # Fine-tune pretrained models
        if self.configs.get_user_cfg()['mode'] == 'fine_tune':
            log.STATUS(sys._getframe().f_lineno,
                       __file__, __name__,
                       "Fine-tuning with best top-tuned model")

            fine_tuned_model = best_model['top_tuned_model'].copy()

            hyper_params = best_model['best_hyper_params'].copy()
            hyper_params['batch_size'] = hyper_params['learning_rate'] = hyper_params['decay_rate_per_10_steps'] = None

            _fit_model(fine_tuned_model, best_model['pretrained_model'],
                       self.arch, self.prep, self.multi_label,
                       epochs, verbose,
                       params, hyper_params,
                       fine_tune=True)

            val_acc = max(fine_tuned_model['history'].history['val_acc'])
            if max_val_acc < val_acc:
                log.STATUS(sys._getframe().f_lineno,
                           __file__, __name__,
                           "Selecting fine-tuned model")
                max_val_acc = val_acc
                final_model = fine_tuned_model.copy()
            else:
                log.STATUS(sys._getframe().f_lineno,
                           __file__, __name__,
                           "Selecting best top-tuned model")
                final_model = best_model['top_tuned_model'].copy()
        else:
            log.STATUS(sys._getframe().f_lineno,
                       __file__, __name__,
                       "Selecting best top-tuned model")
            final_model = best_model['top_tuned_model'].copy()

        # Train embedding models
        if self.connector_trn['data_org'] != 'multi_dir':
            best_emb_model = {}
            max_val_acc = 0
            for emb_model in self.model_selection.emb_models:
                _fit_emb_model(emb_model, self.arch, self.prep, self.configs,
                               self.multi_label, epochs, int(verbose))

                emb_model['val_acc'] = emb_model['model'].score(self.prep.x_val, self.prep.y_val)

                if max_val_acc < emb_model['val_acc']:
                    max_val_acc = emb_model['val_acc']
                    best_emb_model = emb_model.copy()

            if not final_model or max_val_acc > max(final_model['history'].history['val_acc']):
                final_model = best_emb_model.copy()

        # Save best model
        saved_model_dir = self.connector_trn['dir_to_save_model']
        _save_model_details(saved_model_dir, final_model, self.prep, self.arch)

        _cleanup()

        return IXO_RET_SUCCESS, saved_model_dir

def _get_params(kwargs) -> dict:
    params = {}

    params['loss'] = None
    if 'loss' in kwargs:
        params['loss'] = kwargs['loss']

    params['metrics'] = []
    if 'metrics' in kwargs:
        params['metrics'] = kwargs['metrics']
    if not isinstance(params['metrics'], list):
        params['metrics'] = [params['metrics']]

    params['output_layer'] = {}
    if 'output_layer' in kwargs:
        params['output_layer'] = kwargs['output_layer']

    params['max_length'] = None
    if 'max_length' in kwargs:
        params['max_length'] = kwargs['max_length']

    return params

def _permute_hyper_params(kwargs) -> ([list], dict):
    hyper_params_permute = []
    hyper_params_search_space = {}

    batch_size = None
    if 'batch_size' in kwargs:
        batch_size = kwargs['batch_size']
    if not isinstance(batch_size, list):
        batch_size = [batch_size]
    hyper_params_permute.append(batch_size)
    hyper_params_search_space['batch_size'] = batch_size

    optimizer = None
    if 'optimizer' in kwargs:
        optimizer = kwargs['optimizer']
    if not isinstance(optimizer, list):
        optimizer = [optimizer]
    hyper_params_permute.append(optimizer)
    hyper_params_search_space['optimizer'] = optimizer

    learning_rate = None
    if 'learning_rate' in kwargs:
        learning_rate = kwargs['learning_rate']
    if not isinstance(learning_rate, list):
        learning_rate = [learning_rate]
    hyper_params_permute.append(learning_rate)
    hyper_params_search_space['learning_rate'] = learning_rate

    decay_rate_per_10_steps = None
    if 'decay_rate_per_10_steps' in kwargs:
        decay_rate_per_10_steps = kwargs['decay_rate_per_10_steps']
    if not isinstance(decay_rate_per_10_steps, list):
        decay_rate_per_10_steps = [decay_rate_per_10_steps]
    hyper_params_permute.append(decay_rate_per_10_steps)
    hyper_params_search_space['decay_rate_per_10_steps'] = decay_rate_per_10_steps

    hidden_layers = ()
    if 'hidden_layers' in kwargs:
        hidden_layers = kwargs['hidden_layers']
    if not isinstance(hidden_layers, list):
        hidden_layers = [hidden_layers]
    hyper_params_permute.append(hidden_layers)
    hyper_params_search_space['hidden_layers'] = hidden_layers

    hyper_params_permute = list(itertools.product(*hyper_params_permute))
    return hyper_params_permute, hyper_params_search_space

def _tune_hyper_params(pretrained_model:dict,
                       params:dict, hyper_params_permute:list,
                       arch:object, prep:object,
                       multi_label:bool, epochs:int, verbose:bool) -> (dict, dict):
    min_val_loss = np.Inf
    best_model = None
    best_hyper_params = None

    top_tuned_model = pretrained_model.copy()
    top_tuned_model['model'] = None
    top_tuned_model['model_arch'] = None

    for item in hyper_params_permute:
        hyper_params = {}
        hyper_params['batch_size'], hyper_params['optimizer'], \
        hyper_params['learning_rate'], hyper_params['decay_rate_per_10_steps'],\
        hyper_params['hidden_layers'] = item

        _fit_model(top_tuned_model, pretrained_model,
                   arch, prep, multi_label,
                   epochs, verbose,
                   params, hyper_params,
                   fine_tune=False)

        val_loss = min(top_tuned_model['history'].history['val_loss'])
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            best_model = top_tuned_model.copy()
            best_hyper_params = hyper_params.copy()

    return best_model, best_hyper_params

def _fit_model(model:dict, pretrained_model:dict,
               arch:object, prep:object, multi_label:bool,
               epochs:int, verbose:bool,
               params:dict, hyper_params:dict,
               fine_tune:bool):
    model['model'], model['model_arch'] =\
        arch.create_model(pretrained_model,
                          is_trn=True,
                          fine_tune=fine_tune,
                          source=model['source'],
                          multi_label=multi_label,
                          target_one_hot_encoded=prep.target_one_hot_encoded,
                          params=params,
                          hyper_params=hyper_params)

    if model['model']:
        if prep.connector['data_org'] == 'multi_dir':
            prep.gen_multi_dir_data(hyper_params['batch_size'])
            batch_size = None
        else:
            batch_size = hyper_params['batch_size']

        x_trn, y_trn = arch.tokenize_input(model['model_arch'],
                                           prep.x_trn, prep.y_trn,
                                           model['prep_model'])
        x_val, y_val = arch.tokenize_input(model['model_arch'],
                                           prep.x_val, prep.y_val,
                                           model['prep_model'])

        model['history'] = model['model'].fit(x_trn, y_trn,
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              verbose=verbose,
                                              class_weight=prep.class_weight,
                                              validation_data=(x_val, y_val))

        log.STATUS(sys._getframe().f_lineno,
                   __file__, __name__,
                   "{} history: {}".format(model['name'],
                                           model['history'].history))

def _fit_emb_model(emb_model:dict,
                   arch:object, prep:object, configs:object,
                   multi_label:bool, epochs:int, verbose:int):
    emb_pipeline, emb_model['model_arch'] = arch.create_emb_pipeline(emb_model, prep.class_weight,
                                                                     prep.target_one_hot_encoded, multi_label, verbose)

    if multi_label or configs.get_user_cfg()['mode'] == 'emb_fit':
        _ = emb_pipeline.fit(prep.x_trn, prep.y_trn)
        emb_model['model'] = emb_pipeline
    else:
        grid_search = GridSearchCV(emb_pipeline, emb_model['hyper_params'],
                                   scoring='f1_weighted',
                                   cv=min(epochs, configs.get_default_arch_param_settings('cv')),
                                   verbose=verbose)
        _ = grid_search.fit(prep.x_trn, prep.y_trn)
        emb_model['model'] = grid_search.best_estimator_
        emb_model['best_hyper_params'] = emb_model['model'].get_params()

def _save_model_details(saved_model_dir:str, best_model:object, prep:object, arch:object):
    saved_model_details = {}
    saved_model_details['saved_prep_paths'] = prep.save_prep_data_details()

    save_weights_only = False
    if best_model['source'] == 'hfhub':
        save_weights_only = True

    saved_model_details['saved_arch_paths'] = arch.save_arch_details(best_model, saved_model_dir, save_weights_only)
    saved_model_details_path = saved_model_dir + 'saved_model_details.joblib'
    dump(saved_model_details, saved_model_details_path)

def _cleanup():
    gc.collect()

if __name__ == '__main__':
    pass

