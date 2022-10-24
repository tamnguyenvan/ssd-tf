#######################################################################################################################
# Model selection helper for text classifier.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
####################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log

import sys, spacy
from zeugma.embeddings import EmbeddingTransformer

import tensorflow as tf
import tensorflow_text as _
import tensorflow_hub as hub
from transformers import AutoTokenizer, TFAutoModel
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from lightgbm import LGBMClassifier

log = Log()

class TextClfSelectModel():
    def __init__(self, configs:object, list_settings:list, *args, **kwargs):
        self.configs = configs
        self.settings = {}
        self.pretrained_models, self.emb_models = _select_candidate_models(self.configs, self.settings, list_settings)

# Load candidate pretrained models to be fine-tuned and evaluated for this training task before selecting the best model
def _select_candidate_models(configs:object, settings:dict, list_settings:[str]=[]) -> ([dict], [dict]):
    if list_settings:
        _set_settings(settings, list_settings)

    candidate_models = _get_candidate_models(configs, settings)
    pretrained_models, emb_moodels = load_candidate_models(candidate_models)

    return pretrained_models, emb_moodels

# User-specified settings for types of pretrained models to select as candidate models
def _set_settings(settings:dict, list_settings:[str]):
    if 'multi_lang' in list_settings and 'indian_lang' in list_settings:
        log.WARNING(sys._getframe().f_lineno,
                    __file__, __name__,
                    "Models that support Indian languages may not support other non-English languages. \
                    Picking models that support Indian languages")
        list_settings.remove('multi_lang')

    _set_item_in_settings(settings, 'multi_lang', list_settings)
    _set_item_in_settings(settings, 'indian_lang', list_settings)

    _set_item_in_settings(settings, 'light', list_settings)
    _set_item_in_settings(settings, 'very_light', list_settings)

    _set_item_in_settings(settings, 'formal', list_settings)
    _set_item_in_settings(settings, 'lengthy', list_settings)

    _set_item_in_settings(settings, 'small_data', list_settings)
    _set_item_in_settings(settings, 'fast_training', list_settings)

def _set_item_in_settings(settings:dict, item:str, list_settings:[str]):
    settings[item] = False
    if item in list_settings:
        settings[item] = True

def _get_candidate_models(configs:object, settings:dict) -> [dict]:
    if not settings:
        candidate_models = configs.get_models_by_names(configs.get_model_settings('default_models'))
    else:
        mode = lang = size_less_than = size_more_than = type = None

        if settings['fast_training']:
            mode = 'emb_fit'
        elif settings['small_data']:
            mode = 'top_tune'

        user_cfg = configs.get_user_cfg()
        user_cfg['mode'] = mode
        configs.set_user_cfg(user_cfg)

        if settings['multi_lang']:
            lang = 'multi'
        elif settings['indian_lang']:
            lang = 'indian'
        else:
            pass # Try models that support en as well as multi languages

        if settings['light'] and not settings['very_light']:
            size_less_than, size_more_than = _translate_size_settings(configs, 'light_models')
        elif not settings['light'] and settings['very_light']:
            size_less_than, size_more_than = _translate_size_settings(configs, 'very_light_models')
        elif settings['light'] and settings['very_light']:
            size_less_than, _ = _translate_size_settings(configs, 'light_models')
        else:
            pass # Try models of all sizes

        if settings['formal']:
            type = 'cased'
        else:
            pass # Try both cased and uncased models

        candidate_models = configs.get_models_by_attrs(lang=lang,
                                                       size_less_than=size_less_than,
                                                       size_more_than=size_more_than,
                                                       type=type)

        if settings['lengthy'] and not settings['multi_lang']\
                and not settings['indian_lang'] and not settings['light'] and not settings['very_light']:
            candidate_models.append(configs.get_models_by_names(['LongFormer_Large', 'LongFormer_Extra_Large']))

    return candidate_models

def _translate_size_settings(configs:object, setting_name:str) -> (int, int):
    size_less_than = size_more_than = None

    model_settings = configs.get_model_settings(setting_name)
    if 'size_less_than' in model_settings:
        size_less_than = model_settings['size_less_than']
    if 'size_more_than' in model_settings:
        size_more_than = model_settings['size_more_than']

    return size_less_than, size_more_than

def load_candidate_models(candidate_models:[dict]) -> ([dict], [dict]):
    candidate_model_names = [model['name'] for model in candidate_models]
    log.STATUS(sys._getframe().f_lineno,
               __file__, __name__,
               "Candidate models being loaded: {}".format(candidate_model_names))

    pretrained_models = []
    emb_models = []

    for model in candidate_models:
        if model['source'] == 'tfhub' or model['source'] == 'hfhub':
            pretrained_model = model.copy()
            pretrained_model['hub_model'] = None
            pretrained_model['prep_model'] = None

            if pretrained_model['source'] == 'tfhub':
                if pretrained_model['prep_repo_id']:
                    pretrained_model['prep_model'] = hub.KerasLayer(pretrained_model['prep_repo_id'])

                pretrained_model['hub_model'] = hub.KerasLayer(pretrained_model['repo_id'],
                                                               trainable=True)
            elif pretrained_model['source'] == 'hfhub':
                pretrained_model['prep_model'] = AutoTokenizer.from_pretrained(pretrained_model['ckpt'])
                pretrained_model['hub_model'] = TFAutoModel.from_pretrained(pretrained_model['ckpt'])
            else:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__,
                          "Unrecognized source for pretrained model: {}".format(pretrained_model['source']))

            pretrained_models.append(pretrained_model)
        elif model['source'] == 'sklearn':
            emb_model = model.copy()
            emb_model['feat_vec_model'] = None
            emb_model['reduce_dim_model'] = None
            emb_model['clf_model'] = None

            if emb_model['feat_vec'] == 'TfidfVectorizer':
                emb_model['feat_vec_model'] = globals()[emb_model['feat_vec']]
            elif emb_model['feat_vec'] == 'all-mpnet-base-v2' or emb_model['feat_vec'] == 'all-MiniLM-L6-v2':
                emb_model['feat_vec_model'] = SentenceTransformer(emb_model['feat_vec'])
            elif emb_model['feat_vec'] == 'en_core_web_md' or emb_model['feat_vec'] == 'en_core_web_lg':
                emb_model['feat_vec_model'] = spacy.load(emb_model['feat_vec'])
            elif emb_model['feat_vec'] == 'word2vec' or emb_model['feat_vec'] == 'glove'\
                    or emb_model['feat_vec'] == 'fasttext':
                emb_model['feat_vec_model'] = EmbeddingTransformer(emb_model['feat_vec'])
            elif emb_model['feat_vec'] == 'https://tfhub.dev/google/universal-sentence-encoder/4'\
                    or emb_model['feat_vec'] == 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3':
                emb_model['feat_vec_model'] = hub.KerasLayer(emb_model['feat_vec'], trainable=False)
            else:
                log.ERROR(sys._getframe().f_lineno,
                          __file__, __name__,
                          "Unrecognized feature vector for embedding model: {}".format(emb_model['name']))

            if emb_model['reduce_dim'] == 'TruncatedSVD':
                emb_model['reduce_dim_model'] = globals()[emb_model['reduce_dim']]

            emb_model['clf_model'] = globals()[emb_model['clf']]
            if 'clf_stack' in emb_model:
                emb_model['clf_stack_model'] = [globals()[clf] for clf in emb_model['clf_stack']]

            emb_models.append(emb_model)

    return pretrained_models, emb_models

if __name__ == '__main__':
    pass

