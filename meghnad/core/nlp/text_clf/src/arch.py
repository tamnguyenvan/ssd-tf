#######################################################################################################################
# Fine-tuning architecture for text classifier.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
####################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.core.nlp.text_clf.src.select_model import load_candidate_models

import sys
from joblib import dump
import numpy as np

import tensorflow as tf
from transformers import AutoTokenizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier

log = Log()

class TextClfArch():
    def __init__(self, configs:object, num_classes:int, *args, **kwargs):
        self.configs = configs
        self.num_classes = num_classes

    # Create fine-tuning model architecture for pretrained model
    def create_model(self,
                     pretrained_model:dict,
                     is_trn:bool,
                     fine_tune:bool,
                     source:str,
                     multi_label:bool=False,
                     target_one_hot_encoded:bool=False,
                     params:dict={},
                     hyper_params:dict={}) -> (object, dict):
        prep_model = pretrained_model['prep_model']
        hub_model = pretrained_model['hub_model']

        ret = _sanitize_arch_inputs(self.configs, is_trn, fine_tune,
                                    params, hyper_params,
                                    multi_label, target_one_hot_encoded, self.num_classes)
        if ret != IXO_RET_SUCCESS:
            return None, None

        if source == 'tfhub':
            inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)

            if prep_model:
                outputs = hub_model(prep_model(inputs))
            else:
                outputs = hub_model(inputs)
        elif source == 'hfhub':
            input_ids = tf.keras.layers.Input(shape=(params['max_length'], ), dtype=tf.int32)
            attention_mask = tf.keras.layers.Input(shape=(params['max_length'], ), dtype=tf.int32)
            inputs = [input_ids, attention_mask]

            outputs = hub_model({'input_ids': input_ids, 'attention_mask': attention_mask})
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Unrecognized source for model: {}".format(source))

        if pretrained_model['post_proc'] == '0':
            outputs = outputs[0]
        elif pretrained_model['post_proc'] == 'pooled_output':
            outputs = outputs['pooled_output']
        elif pretrained_model['post_proc'] == 'default-l2_norm':
            outputs = outputs['default'] / np.linalg.norm(outputs['default'], 2, axis=1, keepdims=True)
        elif pretrained_model['post_proc'] == 'pooler_output':
            outputs = outputs['pooler_output']
        elif pretrained_model['post_proc'] == 'last_hidden_state':
            outputs = tf.keras.layers.Flatten()(outputs['last_hidden_state'])

        ret, outputs = _build_layers(outputs, hyper_params['hidden_layers'], params['output_layer'])
        if ret != IXO_RET_SUCCESS:
            return None, None

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        if not fine_tune:
            freeze_until = len(model.layers) - (len(hyper_params['hidden_layers']) + 1)
            count_layers = 0
            for layer in model.layers:
                layer.trainable = False
                count_layers += 1
                if count_layers >= freeze_until:
                    break

        if (is_trn and log.log_level >= IXO_LOG_STATUS) or (not is_trn and log.log_level >= IXO_LOG_VERBOSE):
            model.summary()

        if hyper_params['decay_rate_per_10_steps']:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=hyper_params['learning_rate'],
                                                                decay_steps=10,
                                                                decay_rate=hyper_params['decay_rate_per_10_steps'],
                                                                staircase=True)
        else:
            lr = hyper_params['learning_rate']

        try:
            opt = tf.keras.optimizers.get({'class_name': hyper_params['optimizer'], 'config': {'lr': lr}})
        except:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Optimizer {} not supported".format(hyper_params['optimizer']))
            return None, None

        ret, metrics = _translate_metrics(params['metrics'])
        if ret != IXO_RET_SUCCESS:
            return None, None

        model.compile(optimizer=opt,
                      loss=params['loss'],
                      metrics=metrics)

        if is_trn:
            model_arch = {}

            # Remember arch details
            for key in pretrained_model:
                if not (key == 'hub_model' or key == 'prep_model'):
                    model_arch[key] = pretrained_model[key]

            model_arch['num_classes'] = self.num_classes
            model_arch['multi_label'] = multi_label
            model_arch['target_one_hot_encoded'] = target_one_hot_encoded

            # Remember params
            model_arch['params'] = {}
            model_arch['params']['loss'] = params['loss']
            model_arch['params']['metrics'] = params['metrics']
            model_arch['params']['output_layer'] = params['output_layer']
            model_arch['params']['max_length'] = params['max_length']

            # Remember hyper-params
            model_arch['hyper_params'] = {}
            model_arch['hyper_params']['batch_size'] = hyper_params['batch_size']
            model_arch['hyper_params']['optimizer'] = hyper_params['optimizer']
            model_arch['hyper_params']['learning_rate'] = hyper_params['learning_rate']
            model_arch['hyper_params']['decay_rate_per_10_steps'] = hyper_params['decay_rate_per_10_steps']
            model_arch['hyper_params']['hidden_layers'] = hyper_params['hidden_layers']

            return model, model_arch
        else:
            return model, None

    # Save architecture data details to be used during prediction
    def save_arch_details(self, model:dict, dir_to_save_model:str, save_weights_only:bool=False) -> dict:
        saved_arch_paths = {}

        if model['source'] == 'sklearn':
            saved_arch_paths['model'] = dir_to_save_model + "best_model.joblib"
            dump(model['model'], saved_arch_paths['model'])
        elif not save_weights_only:
            saved_arch_paths['model'] = dir_to_save_model + "best_model/"
            tf.keras.models.save_model(model['model'], saved_arch_paths['model'])
        else:
            saved_arch_paths['model_weights'] = dir_to_save_model + "best_model/"
            model['model'].save_weights(saved_arch_paths['model_weights'])

        saved_arch_paths['model_arch'] = dir_to_save_model + 'model_arch.joblib'
        dump(model['model_arch'], saved_arch_paths['model_arch'])

        log.STATUS(sys._getframe().f_lineno,
                   __file__, __name__,
                   "Architecture details saved: {}".format(saved_arch_paths))

        return saved_arch_paths

    # Load architecture data details saved during training
    def load_arch_details(self, model_arch:dict, model_weights_path:str) -> object:
        candidate_models = self.configs.get_models_by_names([model_arch['name']])
        pretrained_model, _ = load_candidate_models(candidate_models)
        pretrained_model = pretrained_model[0]

        model, _ = self.create_model(pretrained_model,
                                     is_trn=False,
                                     fine_tune=False,
                                     source=model_arch['source'],
                                     multi_label=model_arch['multi_label'],
                                     target_one_hot_encoded=model_arch['target_one_hot_encoded'],
                                     params=model_arch['params'],
                                     hyper_params=model_arch['hyper_params'])
        model.load_weights(model_weights_path)

        return model

    # Tokenize input (relevant only for models loaded from hfhub)
    def tokenize_input(self, model_arch:dict,
                       sequences:[str], targets:[int]=None,
                       tokenizer:object=None) -> object:
        if model_arch['source'] == 'hfhub':
            max_length = model_arch['params']['max_length']

            if not tokenizer:
                candidate_model = self.configs.get_models_by_names([model_arch['name']])[0]
                tokenizer = AutoTokenizer.from_pretrained(candidate_model['ckpt'])

            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            if isinstance(sequences, tf.data.Dataset):
                for data_elems in sequences.take(1):
                    sequences = data_elems[0].numpy().tolist()
                    targets = data_elems[1]
            elif not isinstance(sequences, list):
                sequences = sequences.values.tolist()

            sequences = [str(sequence) for sequence in sequences]

            tokens = tokenizer(sequences,
                               max_length=max_length,
                               padding='max_length',
                               add_special_tokens=True,
                               truncation=True,
                               return_tensors="tf")
            return [tokens.input_ids, tokens.attention_mask], targets
        else:
            return sequences, targets

    class _TextClfArchEmbFeatVec(BaseEstimator, TransformerMixin):
        def __init__(self, feat_vec_model: object, feat_vec: str):
            self.feat_vec = feat_vec
            self.feat_vec_model = feat_vec_model

        def fit(self, X, y):
            return self

        def transform(self, X):
            if self.feat_vec == 'word2vec' or self.feat_vec == 'glove' or self.feat_vec == 'fasttext':
                return [self.feat_vec_model.transform(text) for text in X]
            elif self.feat_vec == 'en_core_web_md' or self.feat_vec == 'en_core_web_lg':
                return [self.feat_vec_model(text).vector for text in X]
            elif self.feat_vec == 'all-mpnet-base-v2' or self.feat_vec == 'all-MiniLM-L6-v2':
                return [self.feat_vec_model.encode(text) for text in X]
            elif self.feat_vec == 'https://tfhub.dev/google/universal-sentence-encoder/4'\
                    or self.feat_vec == 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3':
                return [self.feat_vec_model([text])[0].numpy() for text in X]
            else:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__,
                          "Feature vector {} not supported".format(self.feat_vec))

    # Create pipeline for embedding model
    def create_emb_pipeline(self, emb_model:dict, class_weight:dict=None,
                            target_one_hot_encoded:bool=False, multi_label:bool=False, verbose:int=0) -> object:
        if emb_model['feat_vec'] == 'TfidfVectorizer':
            feat_vec_model = emb_model['feat_vec_model']()
        else:
            feat_vec_model = self._TextClfArchEmbFeatVec(emb_model['feat_vec_model'], emb_model['feat_vec'])

        if emb_model['reduce_dim_model']:
            reduce_dim_model = emb_model['reduce_dim_model']()
        else:
            reduce_dim_model = None

        if not class_weight:
            class_weight = 'balanced'

        if emb_model['clf_model'] == 'MultinomialNB' or emb_model['clf_model'] == 'KNeighborsClassifier':
            clf_model = emb_model['clf_model']()
        elif emb_model['clf_model'] == 'GradientBoostingClassifier':
            clf_model = emb_model['clf_model'](random_state=IXO_SEED, verbose=verbose)
        elif emb_model['clf_model'] == 'StackingClassifier':
            emb_model['clf_model'](stack_model(random_state=IXO_SEED, class_weight=class_weight, verbose=verbose)\
                                   for stack_model in emb_model['clf_stack_model'])
        else:
            clf_model = emb_model['clf_model'](random_state=IXO_SEED, class_weight=class_weight, verbose=verbose)

        if multi_label: #and not emb_model['native_multi_label_support']:
            clf_model = MultiOutputClassifier(clf_model)

        if reduce_dim_model:
            steps = [('feat_vec', feat_vec_model),
                     ('reduce_dim', reduce_dim_model),
                     ('clf', clf_model)]
        else:
            steps = [('feat_vec', feat_vec_model),
                     ('clf', clf_model)]

        emb_pipeline = Pipeline(steps=steps)

        model_arch = {}
        model_arch['source'] = emb_model['source']
        model_arch['num_classes'] = self.num_classes
        model_arch['multi_label'] = multi_label
        model_arch['target_one_hot_encoded'] = target_one_hot_encoded

        return emb_pipeline, model_arch

# Verify and rectify (if needed) the architectural inputs
def _sanitize_arch_inputs(configs:object, is_trn:bool, fine_tune:bool,
                          params:dict, hyper_params:dict,
                          multi_label:bool, target_one_hot_encoded:bool, num_classes:int) -> int:
    if is_trn:
        # Get params
        if not params or not 'loss' in params or not params['loss']:
            params['loss'] = configs.get_default_arch_param_settings('loss')
        if not params or not 'metrics' in params or not params['metrics']:
            params['metrics'] = configs.get_default_arch_param_settings('metrics')
        if not params or not 'output_layer' in params or not params['output_layer']:
            params['output_layer'] = configs.get_default_arch_param_settings('output_layer')
        if not params or not 'max_length' in params or not params['max_length']:
            params['max_length'] = configs.get_default_arch_param_settings('max_length')

        ret = _sanitize_params(params, multi_label, target_one_hot_encoded)
        if ret != IXO_RET_SUCCESS:
            return ret

        # Get hyper-params
        if not hyper_params or not 'batch_size' in hyper_params or not hyper_params['batch_size']:
            hyper_params['batch_size'] = configs.get_default_arch_hyper_param_settings('batch_size')

        if not hyper_params or not 'optimizer' in hyper_params or not hyper_params['optimizer']:
            hyper_params['optimizer'] = configs.get_default_arch_hyper_param_settings('optimizer')

        if not hyper_params or not 'learning_rate' in hyper_params or not hyper_params['learning_rate']:
            if fine_tune:
                hyper_params['learning_rate'] = configs.get_default_arch_hyper_param_settings('learning_rate')['fine_tune']
            else:
                hyper_params['learning_rate'] = configs.get_default_arch_hyper_param_settings('learning_rate')['top_tune']

        if not hyper_params or not 'decay_rate_per_10_steps' in hyper_params\
                or hyper_params['decay_rate_per_10_steps'] is None:  # decay_rate_per_10_steps must be 0.0 if no decay is expected
            hyper_params['decay_rate_per_10_steps'] = configs.get_default_arch_hyper_param_settings('decay_rate_per_10_steps')

        if not hyper_params or not 'hidden_layers' in hyper_params or not hyper_params['hidden_layers']:
            hyper_params['hidden_layers'] = configs.get_default_arch_hyper_param_settings('hidden_layers')

    # Following sanitization steps are needed during training as well as prediction
    if 'units' in params['output_layer'] and params['output_layer']['units'] != num_classes:
        log.WARNING(sys._getframe().f_lineno,
                    __file__, __name__,
                    "Changing number of units in output layer to be same as number of classes")
    params['output_layer']['units'] = num_classes

    try:
        # Check if hyper_params['hidden_layers'] is only a tuple, and not a list of tuples (i.e., only 1 hidden layer)
        _ = hyper_params['hidden_layers'][1]
    except:
        # And if so, convert it into a list of tuples for ease of downstream processing
        hyper_params['hidden_layers'] = [hyper_params['hidden_layers']]

    return IXO_RET_SUCCESS

# Verify and rectify (if needed) the parameters
def _sanitize_params(params:dict,
                     multi_label:bool,
                     target_one_hot_encoded:bool) -> (int, str, dict):
    if target_one_hot_encoded:
        if params['loss'] == 'sparse_categorical_crossentropy':
            log.WARNING(sys._getframe().f_lineno,
                        __file__, __name__,
                        "Sparse categorical crossentropy is not appropriate for one-hot encoded target.\
                        Changing to categorical crossentropy instead")
            params['loss'] = 'categorical_crossentropy'
    else:
        if params['loss'] == 'categorical_crossentropy':
            log.WARNING(sys._getframe().f_lineno,
                        __file__, __name__,
                        "Categorical crossentropy is not appropriate for label encoded target.\
                        Changing to sparse categorical crossentropy instead")
            params['loss'] = 'sparse_categorical_crossentropy'
        elif params['loss'] == 'binary_crossentropy':
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Invalid inputs. Target needs to be one-hot encoded for binary cross entropy")
            return IXO_RET_INVALID_INPUTS

    if multi_label:
        if not target_one_hot_encoded:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Invalid inputs. Target needs to be one-hot encoded for multi label classification")
            return IXO_RET_INVALID_INPUTS

        if params['output_layer']['activation'] == 'softmax':
            log.WARNING(sys._getframe().f_lineno,
                        __file__, __name__,
                        "Softmax is not appropriate for multi label classification task.\
                        Changing to sigmoid instead")
            params['output_layer']['activation'] = 'sigmoid'

        if params['loss'] == 'categorical_crossentropy':
            log.WARNING(sys._getframe().f_lineno,
                        __file__, __name__,
                        "Categorical crossentropy is not appropriate for multi label classification task.\
                        Changing to binary crossentropy instead")
            params['loss'] = 'binary_crossentropy'

    return IXO_RET_SUCCESS

def _check_layer_config(supported_params: [str], received_params: [str]):
    ret = IXO_RET_SUCCESS
    params_unsupported = list(set(received_params) - set(supported_params))

    if len(params_unsupported):
        log.ERROR(sys._getframe().f_lineno,
                  __file__, __name__,
                  "Layer params {} not supported".format(params_unsupported))
        ret = IXO_RET_NOT_SUPPORTED

    params_not_received = list(set(supported_params) - set(received_params))
    if len(params_not_received):
        log.ERROR(sys._getframe().f_lineno,
                  __file__, __name__,
                  "Layer params {} not received".format(params_not_received))
        ret = IXO_RET_INVALID_INPUTS

    return ret

def _build_layers(outputs:object, hidden_layers:(dict), output_layer:dict) -> (int, object):
    for layer in hidden_layers:
        if layer['type'] == 'Reshape':
            ret = _check_layer_config(supported_params=['type', 'target_shape'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            target_shape = layer['target_shape']
            outputs = tf.keras.layers.Reshape(target_shape=target_shape)(outputs)
        elif layer['type'] == 'Dense':
            ret = _check_layer_config(supported_params=['type', 'units', 'activation'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            units = layer['units']
            activation = layer['activation']
            outputs = tf.keras.layers.Dense(units=units, activation=activation)(outputs)
        elif layer['type'] == 'LSTM':
            ret = _check_layer_config(supported_params=['type', 'units'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            units = layer['units']
            outputs = tf.keras.layers.LSTM(units=units)(outputs)
        elif layer['type'] == 'Bidirectional_LSTM':
            ret = _check_layer_config(supported_params=['type', 'units', 'return_sequences'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            units = layer['units']
            return_sequences = layer['return_sequences']
            outputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=units,
                                                                         return_sequences=return_sequences))(outputs)
        elif layer['type'] == 'GRU':
            ret = _check_layer_config(supported_params=['type', 'units'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            units = layer['units']
            outputs = tf.keras.layers.GRU(units=units)(outputs)
        elif layer['type'] == 'Attention':
            ret = _check_layer_config(supported_params=['type'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            outputs = tf.keras.layers.Attention()(outputs)
        elif layer['type'] == 'Dropout':
            ret = _check_layer_config(supported_params=['type', 'rate'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            rate = layer['rate']
            outputs = tf.keras.layers.Dropout(rate=rate)(outputs)
        elif layer['type'] == 'Conv1D':
            ret = _check_layer_config(supported_params=['type', 'filters', 'kernel_size',
                                                        'strides', 'activation'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            filters = layer['filters']
            kernel_size = layer['kernel_size']
            strides = layer['strides']
            activation = layer['activation']
            outputs = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                             strides=strides, activation=activation)(outputs)
        elif layer['type'] == 'Conv2D':
            ret = _check_layer_config(supported_params=['type', 'filters', 'kernel_size',
                                                        'strides', 'activation'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            filters = layer['filters']
            kernel_size = layer['kernel_size']
            strides = layer['strides']
            activation = layer['activation']
            outputs = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                             strides=strides, activation=activation)(outputs)
        elif layer['type'] == 'Conv3D':
            ret = _check_layer_config(supported_params=['type', 'filters', 'kernel_size',
                                                        'strides', 'activation'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            filters = layer['filters']
            kernel_size = layer['kernel_size']
            strides = layer['strides']
            activation = layer['activation']
            outputs = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size,
                                             strides=strides, activation=activation)(outputs)
        elif layer['type'] == 'GlobalMaxPool1D':
            ret = _check_layer_config(supported_params=['type'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            outputs = tf.keras.layers.GlobalMaxPool1D()(outputs)
        elif layer['type'] == 'GlobalAveragePooling1D':
            ret = _check_layer_config(supported_params=['type'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
        elif layer['type'] == 'MaxPool1D':
            ret = _check_layer_config(supported_params=['type', 'pool_size', 'strides'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            pool_size = layer['pool_size']
            strides = layer['strides']
            outputs = tf.keras.layers.MaxPool1D(pool_size=pool_size, strides=strides)(outputs)
        elif layer['type'] == 'AveragePooling1D':
            ret = _check_layer_config(supported_params=['type', 'pool_size', 'strides'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            pool_size = layer['pool_size']
            strides = layer['strides']
            outputs = tf.keras.layers.AveragePooling1D(pool_size=pool_size, strides=strides)(outputs)
        elif layer['type'] == 'Flatten':
            ret = _check_layer_config(supported_params=['type'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            outputs = tf.keras.layers.Flatten()(outputs)
        elif layer['type'] == 'BatchNormalization':
            ret = _check_layer_config(supported_params=['type'],
                                      received_params=list(set(layer.keys())))
            if ret != IXO_RET_SUCCESS:
                return ret, None

            outputs = tf.keras.layers.BatchNormalization()(outputs)
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Layer type {} not supported as hidden layer".format(layer['type']))
            return IXO_RET_NOT_SUPPORTED, None

    if output_layer['type'] == 'Dense':
        ret = _check_layer_config(supported_params=['type', 'units', 'activation'],
                                  received_params=list(set(output_layer.keys())))
        if ret != IXO_RET_SUCCESS:
            return ret, None

        units = output_layer['units']
        activation = output_layer['activation']
        outputs = tf.keras.layers.Dense(units=units, activation=activation)(outputs)
    else:
        log.ERROR(sys._getframe().f_lineno,
                  __file__, __name__,
                  "Layer type {} not supported as output layer".format(layer['type']))
        return IXO_RET_NOT_SUPPORTED, None

    return IXO_RET_SUCCESS, outputs

def _translate_metrics(metrics:[str]) -> [object]:
    metrics_instances = []
    ret = IXO_RET_SUCCESS

    for metric in metrics:
        if str(metric).casefold() == 'accuracy' or str(metric).casefold() == 'acc':
            metrics_instances.append('acc')
        #elif str(metric).casefold() == 'auc':
        #    metrics_instances.append(tf.keras.metrics.AUC())
        #elif str(metric).casefold() == 'iou':
        #    metrics_instances.append(tf.keras.metrics.IoU())
        else:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Metrics {} not supported".format(metric))
            ret = IXO_RET_NOT_SUPPORTED

    return ret, metrics_instances

if __name__ == '__main__':
    pass

