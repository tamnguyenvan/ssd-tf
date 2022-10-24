#######################################################################################################################
# Configurations for NLP classifier.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.cfg.config import MeghnadConfig

_text_clf_cfg =\
{
    'models':
    {
        'USE_en':
        {
            'name': 'USE_en',
            'repo_id': 'https://tfhub.dev/google/universal-sentence-encoder/4',
            'prep_repo_id': None,
            'post_proc': None,
            'source': 'tfhub',
            'size_mb': 916,
            'lang': 'en',
            'type': 'uncased',
        },
        'USE_multi':
        {
            'name': 'USE_multi',
            'repo_id': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3',
            'prep_repo_id': None,
            'post_proc': None,
            'source': 'tfhub',
            'size_mb': 245,
            'lang': 'multi',
            'type': 'uncased',
        },
        'USE_multi_large':
        {
            'name': 'USE_multi_large',
            'repo_id': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3',
            'prep_repo_id': None,
            'post_proc': None,
            'source': 'tfhub',
            'size_mb': 303,
            'lang': 'multi',
            'type': 'uncased',
        },
        'st5-base':
        {
            'name': 'st5-base',
            'repo_id': 'https://tfhub.dev/google/sentence-t5/st5-base/1',
            'prep_repo_id': None,
            'post_proc': '0',
            'source': 'tfhub',
            'size_mb': 188,
            'lang': 'en',
            'type': 'uncased',
        },
        'st5-large':
        {
            'name': 'st5-large',
            'repo_id': 'https://tfhub.dev/google/sentence-t5/st5-large/1',
            'prep_repo_id': None,
            'post_proc': '0',
            'source': 'tfhub',
            'size_mb': 569,
            'lang': 'en',
            'type': 'uncased',
        },
        'st5-3b':
        {
            'name': 'st5-3b',
            'repo_id': 'https://tfhub.dev/google/sentence-t5/st5-3b/1',
            'prep_repo_id': None,
            'post_proc': '0',
            'source': 'tfhub',
            'size_mb': 2050,
            'lang': 'en',
            'type': 'uncased',
        },
        'electra_small':
        {
            'name': 'electra_small',
            'repo_id': 'https://tfhub.dev/google/electra_small/2',
            'prep_repo_id': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 48,
            'lang': 'en',
            'type': 'uncased',
        },
        'electra_large':
        {
            'name': 'electra_large',
            'repo_id': 'https://tfhub.dev/google/electra_large/2',
            'prep_repo_id': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 1160,
            'lang': 'en',
            'type': 'uncased',
        },
        'LaBSE':
        {
            'name': 'LaBSE',
            'repo_id': 'https://tfhub.dev/google/LaBSE/2',
            'prep_repo_id': 'https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2',
            'post_proc': 'default-l2_norm',
            'source': 'tfhub',
            'size_mb': 1630,
            'lang': 'multi',
            'type': 'uncased',
        },
        'MuRIL':
        {
            'name': 'MuRIL',
            'repo_id': 'https://tfhub.dev/google/MuRIL/1',
            'prep_repo_id': 'https://tfhub.dev/google/MuRIL_preprocess/1',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 842,
            'lang': 'indian',
            'type': 'uncased',
        },
        'MuRIL-Large':
        {
            'name': 'MuRIL-Large',
            'repo_id': 'https://tfhub.dev/google/MuRIL-Large/1',
            'prep_repo_id': 'https://tfhub.dev/google/MuRIL_preprocess/1',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 842,
            'lang': 'indian',
            'type': 'uncased',
        },
        'bert_en_cased_small':
        {
            'name': 'bert_en_cased_small',
            'repo_id': 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4',
            'prep_repo_id': 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 386,
            'lang': 'en',
            'type': 'cased',
        },
        'bert_en_cased_large':
        {
            'name': 'bert_en_cased_large',
            'repo_id': 'https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/4',
            'prep_repo_id': 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 1160,
            'lang': 'en',
            'type': 'cased',
        },
        'bert_en_uncased_small':
        {
            'name': 'bert_en_uncased_small',
            'repo_id': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4',
            'prep_repo_id': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 390,
            'lang': 'en',
            'type': 'uncased',
        },
        'bert_en_uncased_large':
        {
            'name': 'bert_en_uncased_large',
            'repo_id': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4',
            'prep_repo_id': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 1160,
            'lang': 'en',
            'type': 'uncased',
        },
        'bert_multi_cased':
        {
            'name': 'bert_multi_cased',
            'repo_id': 'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4',
            'prep_repo_id': 'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 633,
            'lang': 'multi',
            'type': 'cased',
        },
        'albert_en_large':
        {
            'name': 'albert_en_large',
            'repo_id': 'https://tfhub.dev/tensorflow/albert_en_large/3',
            'prep_repo_id': 'http://tfhub.dev/tensorflow/albert_en_preprocess/3',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 64,
            'lang': 'en',
            'type': 'uncased',
        },
        'albert_en_xlarge':
        {
            'name': 'albert_en_xlarge',
            'repo_id': 'https://tfhub.dev/tensorflow/albert_en_xlarge/3',
            'prep_repo_id': 'http://tfhub.dev/tensorflow/albert_en_preprocess/3',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 210,
            'lang': 'en',
            'type': 'uncased',
        },
        'albert_en_xxlarge':
        {
            'name': 'albert_en_xxlarge',
            'repo_id': 'https://tfhub.dev/tensorflow/albert_en_xxlarge/3',
            'prep_repo_id': 'http://tfhub.dev/tensorflow/albert_en_preprocess/3',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 790,
            'lang': 'en',
            'type': 'uncased',
        },
        'mobilebert_en_uncased':
        {
            'name': 'mobilebert_en_uncased',
            'repo_id': 'https://tfhub.dev/tensorflow/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT/1',
            'prep_repo_id': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 134,
            'lang': 'en',
            'type': 'uncased',
        },
        'mobilebert_multi_cased':
        {
            'name': 'mobilebert_multi_cased',
            'repo_id': 'https://tfhub.dev/tensorflow/mobilebert_multi_cased_L-24_H-128_B-512_A-4_F-4_OPT/1',
            'prep_repo_id': 'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
            'post_proc': 'pooled_output',
            'source': 'tfhub',
            'size_mb': 297,
            'lang': 'multi',
            'type': 'cased',
        },
        'funnel-transformer_small':
        {
            'name': 'funnel-transformer_small',
            'ckpt': 'funnel-transformer/small',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 524,
            'lang': 'en',
            'type': 'uncased',
        },
        'funnel-transformer_large':
        {
            'name': 'funnel-transformer_large',
            'ckpt': 'funnel-transformer/large',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 1550,
            'lang': 'en',
            'type': 'uncased',
        },
        'longformer-base':
        {
            'name': 'longformer-base',
            'ckpt': 'allenai/longformer-base-4096',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 765,
            'lang': 'en',
            'type': 'uncased',
        },
        'longformer-large':
        {
            'name': 'longformer-large',
            'ckpt': 'allenai/longformer-large-4096',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 1970,
            'lang': 'en',
            'type': 'uncased',
        },
        'distilbert-base-en_uncased':
        {
            'name': 'distilbert-base-en_uncased',
            'ckpt': 'distilbert-base-uncased',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 363,
            'lang': 'en',
            'type': 'uncased',
        },
        'distilbert-base-en_cased':
        {
            'name': 'distilbert-base-en_cased',
            'ckpt': 'distilbert-base-cased',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 363,
            'lang': 'en',
            'type': 'cased',
        },
        'distilbert-base-multi_cased':
        {
            'name': 'distilbert-base-multi_cased',
            'ckpt': 'distilbert-base-multilingual-cased',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 911,
            'lang': 'multi',
            'type': 'cased',
        },
        'roberta-base':
        {
            'name': 'roberta-base',
            'ckpt': 'roberta-base',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 657,
            'lang': 'en',
            'type': 'cased',
        },
        'roberta-large':
        {
            'name': 'roberta-large',
            'ckpt': 'roberta-large',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 1630,
            'lang': 'en',
            'type': 'cased',
        },
        'bart-base':
        {
            'name': 'bart-base',
            'ckpt': 'facebook/bart-base',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 558,
            'lang': 'en',
            'type': 'cased',
        },
        'bart-large':
        {
            'name': 'bart-large',
            'ckpt': 'facebook/bart-large',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 1630,
            'lang': 'en',
            'type': 'cased',
        },
        'xlnet-base-cased':
        {
            'name': 'xlnet-base-cased',
            'ckpt': 'xlnet-base-cased',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 565,
            'lang': 'en',
            'type': 'cased',
        },
        'gpt2':
        {
            'name': 'gpt2',
            'ckpt': 'gpt2',
            'post_proc': 'last_hidden_state',
            'source': 'hfhub',
            'size_mb': 498,
            'lang': 'en',
            'type': 'cased',
        },
    },
    'emb_models':
    {
        'tfidf_lin_svc':
        {
            'name': 'tfidf_lin_svc',
            'feat_vec': 'TfidfVectorizer',
            'reduce_dim': None,
            'clf': 'LinearSVC',
            'native_multi_label_support': False,
            'source': 'sklearn',
            'size_mb': 0,
            'lang': 'en',
            'type': 'uncased',
            'hyper_params':
            {
                'feat_vec__lowercase': (True,),
                'feat_vec__stop_words': ('english',),
                'feat_vec__ngram_range': ((1,3),),
                'feat_vec__norm': ('l1', 'l2'),
                'clf__C': (0.1, 1.0),
            },
        },
        'bow_md_lr':
        {
            'name': 'bow_md_lr',
            'feat_vec': 'en_core_web_md',
            'reduce_dim': None,
            'clf': 'LogisticRegression',
            'native_multi_label_support': False,
            'source': 'sklearn',
            'size_mb': 40,
            'lang': 'en',
            'type': 'cased',
            'hyper_params':
            {
                'clf__C': (0.1, 1.0),
            },
        },
        'bow_lg_svc':
        {
            'name': 'bow_lg_svc',
            'feat_vec': 'en_core_web_lg',
            'reduce_dim': 'TruncatedSVD',
            'clf': 'SVC',
            'native_multi_label_support': False,
            'source': 'sklearn',
            'size_mb': 560,
            'lang': 'en',
            'type': 'cased',
            'hyper_params':
            {
                'reduce_dim__n_components': (64,),
                'clf__C': (0.1, 1.0),
            },
        },
        'word2vec_nb':
        {
            'name': 'word2vec_nb',
            'feat_vec': 'word2vec',
            'reduce_dim': None,
            'clf': 'MultinomialNB',
            'native_multi_label_support': False,
            'source': 'sklearn',
            'size_mb': 0,
            'lang': 'en',
            'type': 'cased',
            'hyper_params':
            {
                'clf__alpha': (0.0, 1.0),
            },
        },
        'glove_lgbm':
        {
            'name': 'glove_lgbm',
            'feat_vec': 'glove',
            'reduce_dim': None,
            'clf': 'LGBMClassifier',
            'native_multi_label_support': False,
            'source': 'sklearn',
            'size_mb': 105,
            'lang': 'en',
            'type': 'cased',
            'hyper_params':
            {
                'clf__n_estimators': (10, 100, 500),
                'clf__min_child_samples': (5, 10, 25),
                'clf__learning_rate': (0.001, 0.01, 0.1),
            },
        },
        'fasttext_knn':
        {
            'name': 'fasttext_knn',
            'feat_vec': 'fasttext',
            'reduce_dim': None,
            'clf': 'KNeighborsClassifier',
            'native_multi_label_support': True,
            'source': 'sklearn',
            'size_mb': 958,
            'lang': 'en',
            'type': 'cased',
            'hyper_params':
            {
                'clf__n_neighbors': (5, 10),
            },
        },
        'sent_trf_rf':
        {
            'name': 'sent_trf_rf',
            'feat_vec': 'all-MiniLM-L6-v2',
            'reduce_dim': None,
            'clf': 'RandomForestClassifier',
            'native_multi_label_support': True,
            'source': 'sklearn',
            'size_mb': 91,
            'lang': 'en',
            'type': 'cased',
            'hyper_params':
            {
                'clf__n_estimators': (10, 100),
                'clf__min_samples_leaf': (4, 10),
            },
        },
        'sent_trf_sgd':
        {
            'name': 'sent_trf_sgd',
            'feat_vec': 'all-mpnet-base-v2',
            'reduce_dim': None,
            'clf': 'SGDClassifier',
            'native_multi_label_support': False,
            'source': 'sklearn',
            'size_mb': 438,
            'lang': 'en',
            'type': 'cased',
            'hyper_params':
            {
                'clf__loss': ('perceptron',),
                'clf__max_iter': (10,),
                'clf__alpha': (0.001, 0.0001, 0.00001),
                'clf__penalty': ('l2', 'elasticnet'),
            },
        },
        'use_stack':
        {
            'name': 'use_stack',
            'feat_vec': 'https://tfhub.dev/google/universal-sentence-encoder/4',
            'reduce_dim': 'TruncatedSVD',
            'clf': 'StackingClassifier',
            'native_multi_label_support': False,
            'clf_stack': ['LinearSVC', 'LogisticRegression', 'SGDClassifier'],
            'source': 'sklearn',
            'size_mb': 916,
            'lang': 'en',
            'type': 'uncased',
            'hyper_params':
            {
                'reduce_dim__n_components': (64,),
            },
        },
        'use_multi_gb':
        {
            'name': 'use_multi_gb',
            'feat_vec': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3',
            'reduce_dim': None,
            'clf': 'GradientBoostingClassifier',
            'native_multi_label_support': True,
            'source': 'sklearn',
            'size_mb': 245,
            'lang': 'multi',
            'type': 'uncased',
            'hyper_params':
            {
                'clf__n_estimators': (10, 100, 500),
                'clf__min_samples_leaf': (3, 10, 25),
                'clf__learning_rate': (0.001, 0.01, 0.1),
            },
        },
    },
    'model_settings':
    {
        'default_models': ['USE_multi_large',
                           #'USE_en',
                           #'bert_en_uncased_small',
                           #'roberta-base',
                           #'funnel-transformer_small',
                           #'longformer-base',
                           #'bart-base',
                           #'gpt2',
                           'distilbert-base-en_uncased',
                           'sent_trf_rf',],
        'light_models':
        {
            'size_less_than': 500,
            'size_more_than': 100,
        },
        'very_light_models':
        {
            'size_less_than': 100,
        },
    },
    'arch_settings':
    {
        'default_params':
        {
            'loss': 'sparse_categorical_crossentropy',
            'metrics': ['accuracy'],
            'output_layer':
            {
                'type': 'Dense',
                'activation': 'softmax',
            },
            'max_length': 128,
            'cv': 5,
        },
        'default_hyper_params':
        {
            'batch_size': 8,
            'optimizer': 'adam',
            'learning_rate':
            {
                'top_tune': 1e-3,
                'fine_tune': 1e-5,
            },
            'decay_rate_per_10_steps': 0.9,
            'hidden_layers':
            (
                {
                    'type': 'Dense',
                    'units': 64,
                    'activation': 'relu',
                },
                {
                    'type': 'Dropout',
                    'rate': 0.1,
                },
            ),
        },
    },
}

class TextClfConfig(MeghnadConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.user_cfg = {}
        self.user_cfg['mode'] = 'fine_tune'

    def set_user_cfg(self, user_cfg):
        for key in user_cfg:
            self.user_cfg[key] = user_cfg[key]

    def get_user_cfg(self):
        return self.user_cfg

    def get_models_by_names(self, names:[str]=[]) -> [dict]:
        models = []

        if names:
            for name in names:
                if name in _text_clf_cfg['models']:
                    models.append(_text_clf_cfg['models'][name])
                elif name in _text_clf_cfg['emb_models']:
                    models.append(_text_clf_cfg['emb_models'][name])
        try:
            return models.copy()
        except:
            return models

    def get_models_by_attrs(self, **kwargs) -> [dict]:
        names = list(_text_clf_cfg['models'].keys()) + list(_text_clf_cfg['emb_models'].keys())

        if 'source' in kwargs and kwargs['source']:
            names = [name for name in names\
                     if _text_clf_cfg['models'][name]['source'] == str(kwargs['source'])]
            names += [name for name in names\
                     if _text_clf_cfg['emb_models'][name]['source'] == str(kwargs['source'])]
        if 'size_less_than' in kwargs and kwargs['size_less_than']:
            names = [name for name in names\
                           if _text_clf_cfg['models'][name]['size_mb'] < int(kwargs['size_less_than'])]
            names += [name for name in names\
                     if _text_clf_cfg['emb_models'][name]['size_mb'] < int(kwargs['size_less_than'])]
        if 'size_more_than' in kwargs and kwargs['size_more_than']:
            names = [name for name in names\
                           if _text_clf_cfg['models'][name]['size_mb'] > int(kwargs['size_more_than'])]
            names += [name for name in names\
                     if _text_clf_cfg['emb_models'][name]['size_mb'] > int(kwargs['size_more_than'])]
        if 'lang' in kwargs and kwargs['lang']:
            names = [name for name in names\
                           if _text_clf_cfg['models'][name]['lang'] == str(kwargs['lang'])]
            names += [name for name in names\
                     if _text_clf_cfg['emb_models'][name]['lang'] == str(kwargs['lang'])]
        if 'type' in kwargs and kwargs['type']:
            names = [name for name in names\
                           if _text_clf_cfg['models'][name]['type'] == str(kwargs['type'])]
            names += [name for name in names\
                     if _text_clf_cfg['emb_models'][name]['type'] == str(kwargs['type'])]

        models = self.get_models_by_names(names)

        try:
            return models.copy()
        except:
            return models

    # Get settings about various model configuration parameters
    def get_model_settings(self, setting_name:str=None) -> object:
        if setting_name and setting_name in _text_clf_cfg['model_settings']:
            try:
                return _text_clf_cfg['model_settings'][setting_name].copy()
            except:
                return _text_clf_cfg['model_settings'][setting_name]

    # Get default settings about various architectural parameters
    def get_default_arch_param_settings(self, setting_name:str=None) -> object:
        if setting_name and setting_name in _text_clf_cfg['arch_settings']['default_params']:
            try:
                return _text_clf_cfg['arch_settings']['default_params'][setting_name].copy()
            except:
                return _text_clf_cfg['arch_settings']['default_params'][setting_name]

    # Get default settings about various architectural hyper-parameters
    def get_default_arch_hyper_param_settings(self, setting_name:str=None) -> object:
        if setting_name and setting_name in _text_clf_cfg['arch_settings']['default_hyper_params']:
            try:
                return _text_clf_cfg['arch_settings']['default_hyper_params'][setting_name].copy()
            except:
                return _text_clf_cfg['arch_settings']['default_hyper_params'][setting_name]

