#######################################################################################################################
# Training for Topic Model.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Chiranjeevraja
#######################################################################################################################
from utils.log import Log
from utils.common_defs import *
from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.topic_model.cfg.config import TopicModelConfig

import joblib
import os, shutil, sys
import pandas as pd
import numpy as np
import scipy.sparse as ss
import corextopic.corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer

log = Log()

@class_header(
    description='''
    Topic Model training pipeline.''')
class TopicModelTrn():
    def __init__(self):
        tm_config = TopicModelConfig()
        self.parameters = tm_config.get_config()

    @method_header(
        description='''
                Helper for configuring data connectors.''',
        arguments='''
                data_path: location of the training data (point to the file),
                data_type: type of the training data (csv,txt),
                max_keyword_len: to specify the range of n-gram,
                target_cols(optional): attribute names in the data to be used as targets during training,
                anchor_words(optional): anchoring provides a way to guide the topic model towards specific subsets of 
                                        words that the user would like to explore,
                dir_to_save_model(optional): Directory where the model needs to be saved.''')
    def config_connectors(self, data_path:str, data_type:str, max_keyword_len:int, target_cols:str = None, \
                          anchor_words:[str] = [], dir_to_save_model:str = None):
        self.connector_trn = {}
        if data_path:
            self.connector_trn['data_path'] = data_path
        else:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      "Connector not configured with any data")
            return IXO_RET_INCORRECT_CONFIG

        if data_type:
            self.connector_trn['data_type'] = data_type
        else:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      "Connector not configured with a data type")
            return IXO_RET_INCORRECT_CONFIG

        if self.connector_trn['data_type'] not in ['csv','txt']:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__, "Not Implemented")
            return IXO_RET_NOT_IMPLEMENTED

        if (target_cols != None) and (self.connector_trn['data_type'] == 'csv'):
            self.connector_trn['target_cols'] = target_cols
        elif self.connector_trn['data_type'] == 'txt':
            self.connector_trn['target_cols'] = 'None'
        else:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      "Connector not configured with target_cols")

        if max_keyword_len:
            self.connector_trn['max_keyword_len'] = max_keyword_len
        else:
            self.connector_trn['max_keyword_len'] = 1

        if anchor_words:
            self.connector_trn['anchor_words'] = anchor_words
        else:
            self.connector_trn['anchor_words'] = []

        if dir_to_save_model:
            self.connector_trn['dir_to_save_model'] = dir_to_save_model
        else:
            dir_to_save_model = MeghnadConfig().get_meghnad_configs('INT_PATH') + 'topic_model/'
            self.connector_trn['dir_to_save_model'] = dir_to_save_model

        if not os.path.exists(self.connector_trn['dir_to_save_model']):
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__, "Creating directories to save model")
            os.makedirs(self.connector_trn['dir_to_save_model'])
        else:
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                        "Removing existing directory and creating a new one to save model")
            shutil.rmtree(self.connector_trn['dir_to_save_model'])
            os.makedirs(self.connector_trn['dir_to_save_model'])

    @method_header(
        description='''
                preprocessing and training of topic model.''',
        arguments='''
                  ''',
        returns='''
                a 4 member tuple containing the IXO return value, the directory to save model, the 
                topics dictionary and the document list.''')
    def trn_ideas(self) -> (int, str, dict, list):
        # preprocessing
        dl = _get_data_into_list(self.connector_trn['data_path'], self.connector_trn['data_type'], \
                            self.connector_trn['target_cols'])
        anchor_words = self.connector_trn['anchor_words']
        ngram_range = (1,self.connector_trn['max_keyword_len'])
        vectorizer = CountVectorizer(stop_words=self.parameters['stop_words'], \
                                     max_features=self.parameters['max_features'], \
                                     binary=True, ngram_range=ngram_range)
        doc_word = vectorizer.fit_transform(dl)
        doc_word = ss.csr_matrix(doc_word)
        words = list(np.asarray(vectorizer.get_feature_names()))
        not_digit_inds = [ind for ind, word in enumerate(words) if not word.isdigit()]
        doc_word = doc_word[:, not_digit_inds]
        words = [word for ind, word in enumerate(words) if not word.isdigit()]

        # Training
        if anchor_words != []:
            anchored_topic_model = ct.Corex(n_hidden=self.parameters['n_hidden'])
            anchored_topic_model.fit(doc_word, words=words, anchors=anchor_words, \
                                     anchor_strength=self.parameters['anchor_strength'])
            topics = anchored_topic_model.get_topics()
            joblib.dump(anchored_topic_model, self.connector_trn['dir_to_save_model'] + 'anchored_topic_model')
            topic_dict = _get_topic_list(topics)
            return IXO_RET_SUCCESS, self.connector_trn['dir_to_save_model'] + 'anchored_topic_model', topic_dict, dl
        else:
            idea_model_trn = ct.Corex(n_hidden=self.parameters['n_hidden'], words=words)
            idea_model = idea_model_trn.fit(doc_word, words=words)
            topics = idea_model.get_topics()
            joblib.dump(idea_model, self.connector_trn['dir_to_save_model'] + 'idea_model')
            topic_dict = _get_topic_list(topics)
            return IXO_RET_SUCCESS, self.connector_trn['dir_to_save_model'] + 'idea_model', topic_dict, dl


# reads the input file and stores the data in a list
def _get_data_into_list(data_path, data_type, target_cols) -> (list):
    if data_type == 'csv':
        dl = pd.read_csv(data_path, usecols=[target_cols])
        dl = dl[target_cols].values.tolist()
    if data_type == 'txt':
        my_file = open(data_path, "r")
        data = my_file.read()
        dl = data.split(".")
        my_file.close()
    return dl

# converting all topic list to dictionary format
def _get_topic_list(topics) -> (dict):
    topic_list = []
    for k in topics:
        prob_tot = 0
        topic_sublist = []
        for i in range(len(k)):
            prob_tot += k[i][1]
        for i in range(len(k)):
            Second_item = round((k[i][1] / prob_tot) * 100, 2)
            topic_sublist.append((k[i][0], Second_item, k[i][2]))
        topic_list.append(topic_sublist)
    count = 0
    topic_dict = {}
    for i in topic_list:
        corr_lst = []
        inv_corr_lst = []
        for j in i:
            if j[2] == 1:
                corr_lst.append((j[0], j[1]))
            else:
                inv_corr_lst.append((j[0], j[1]))
        topic_dict[count] = {'Corr': corr_lst, 'inv_corr': inv_corr_lst}
        count += 1
    return topic_dict