#######################################################################################################################
# Unit-test for Topic Model.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Chiranjeevraja
#######################################################################################################################

from utils.log import Log
from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig

import unittest
import os, gc
from meghnad.core.nlp.topic_model.src.trn import TopicModelTrn
from meghnad.core.nlp.topic_model.src.pred import TopicModelPred

import warnings
warnings.filterwarnings('ignore')

log = Log()

def _cleanup():
    gc.collect()

def _write_results_tc_1(ret_val, saved_model_dir, results_path, topics):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(results_path + "tc_results.txt", 'w', encoding="utf-8") as f:
        f.write("ret val: ")
        f.write(str(ret_val))
        f.write("\nsaved model dir: ")
        f.write(str(saved_model_dir))
    with open(results_path + "tc_topics.txt", 'w', encoding="utf-8") as f:
        f.write(str(topics))

def _write_results_tc_2(documents_dict, keywords_lst, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(results_path + "tc_keywords_list.txt", 'w', encoding="utf-8") as f:
        f.write(str(keywords_lst))
    with open(results_path + "tc_documents_list.txt", 'w', encoding="utf-8") as f:
        f.write(str(documents_dict))

def _tc_1(testcases_path, results_path):
    data_path = testcases_path + "K8_reviews.csv"
    data_type = 'csv'
    target_cols = "review"
    max_keyword_len = 1
    anchor_words = [["camera","lens","quality"]]

    # training data
    tm_train = TopicModelTrn()
    tm_train.config_connectors(data_path=data_path, data_type=data_type, target_cols=target_cols,\
                               max_keyword_len=max_keyword_len, anchor_words=anchor_words)
    ret_val, directory, topics, docs_list = tm_train.trn_ideas()

    results_path += "tc_1/"
    _write_results_tc_1(ret_val, directory, results_path, topics)

    if ret_val == IXO_RET_SUCCESS:
        return directory, docs_list

def _tc_2(model, docs_list, results_path):
    keyword = "camera"
    top_n = 10

    tm_pred = TopicModelPred(model, keyword, top_n, docs_list)
    documents_dict = tm_pred.get_relevant_docs()
    keywords_lst = tm_pred.get_relevant_keywords()

    results_path += "tc_2/"
    _write_results_tc_2(documents_dict, keywords_lst, results_path)

def _tc_3(testcases_path, results_path):
    data_path = testcases_path + "K8_reviews.csv"
    data_type = 'csv'
    target_cols = "review"
    max_keyword_len = 2
    anchor_words = ["battery issue"]

    # training data
    tm_train = TopicModelTrn()
    tm_train.config_connectors(data_path=data_path, data_type=data_type, target_cols=target_cols,\
                               max_keyword_len=max_keyword_len, anchor_words=anchor_words)
    ret_val, directory, topics, docs_list = tm_train.trn_ideas()

    results_path += "tc_3/"
    _write_results_tc_1(ret_val, directory, results_path, topics)

    if ret_val == IXO_RET_SUCCESS:
        return directory, docs_list

def _tc_4(model, docs_list, results_path):
    keyword = "battery issue"
    top_n = 5

    tm_pred = TopicModelPred(model, keyword, top_n, docs_list)
    documents_dict = tm_pred.get_relevant_docs()
    keywords_lst = tm_pred.get_relevant_keywords()

    results_path += "tc_4/"
    _write_results_tc_2(documents_dict, keywords_lst, results_path)

def _tc_5(testcases_path, results_path):
    data_path = testcases_path + "APPLE_iPhone_SE.csv"
    data_type = 'csv'
    target_cols = "Reviews"
    max_keyword_len = 1

    # training data
    tm_train = TopicModelTrn()
    tm_train.config_connectors(data_path=data_path, data_type=data_type, target_cols=target_cols,\
                               max_keyword_len=max_keyword_len)
    ret_val, directory, topics, docs_list = tm_train.trn_ideas()

    results_path += "tc_5/"
    _write_results_tc_1(ret_val, directory, results_path, topics)

    if ret_val == IXO_RET_SUCCESS:
        return directory, docs_list

def _tc_6(model, docs_list, results_path):
    keyword = "display"
    top_n = 20

    tm_pred = TopicModelPred(model, keyword, top_n, docs_list)
    documents_dict = tm_pred.get_relevant_docs()
    keywords_lst = tm_pred.get_relevant_keywords()

    results_path += "tc_6/"
    _write_results_tc_2(documents_dict, keywords_lst, results_path)

def _tc_7(testcases_path, results_path):
    data_path = testcases_path + "mahindra_zigwheels.txt"
    data_type = 'txt'
    max_keyword_len = 2
    anchor_words = ['engine']

    # training data
    tm_train = TopicModelTrn()
    tm_train.config_connectors(data_path=data_path, data_type=data_type,\
                               max_keyword_len=max_keyword_len, anchor_words=anchor_words)
    ret_val, directory, topics, docs_list = tm_train.trn_ideas()

    results_path += "tc_9/"
    _write_results_tc_1(ret_val, directory, results_path, topics)

    if ret_val == IXO_RET_SUCCESS:
        return directory, docs_list

def _tc_8(model, docs_list, results_path):
    keyword = "engine"
    top_n = 10

    tm_pred = TopicModelPred(model, keyword, top_n, docs_list)
    documents_dict = tm_pred.get_relevant_docs()
    keywords_lst = tm_pred.get_relevant_keywords()

    results_path += "tc_10/"
    _write_results_tc_2(documents_dict, keywords_lst, results_path)

def _perform_tests():
    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/nlp/topic_model/unit-test/"

    results_path = ut_path + "results/"
    testcases_path = ut_path + "testcases/"

    saved_model_dir, docs_list  = _tc_1(testcases_path, results_path)

    _tc_2(saved_model_dir, docs_list, results_path)

    ############################################################################

    saved_model_dir, docs_list = _tc_3(testcases_path, results_path)

    _tc_4(saved_model_dir, docs_list, results_path)

    ###########################################################################

    saved_model_dir, docs_list = _tc_5(testcases_path, results_path)

    _tc_6(saved_model_dir, docs_list, results_path)

    ###########################################################################

    saved_model_dir, docs_list = _tc_7(testcases_path, results_path)

    _tc_8(saved_model_dir, docs_list, results_path)

if __name__ == '__main__':

    _perform_tests()

    unittest.main()

    _cleanup()
