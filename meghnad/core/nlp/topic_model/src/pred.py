#######################################################################################################################
# Prediction for Topic Model.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Chiranjeevraja
#######################################################################################################################
from utils.log import Log
from utils.common_defs import *
from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig

import joblib

log = Log()

@class_header(
    description='''
        Topic Model prediction pipeline.''')
class TopicModelPred():
    def __init__(self, saved_model_dir, keyword, top_n, docs_list):
        self.model = joblib.load(saved_model_dir)
        self.keyword = keyword
        self.top_n = top_n
        self.docs_list = docs_list

    @method_header(
        description='''
                Find documents, which is similar to the keyword.''',
        returns='''
                a dictionary which contains document id with document which is similar to the keyword.''')
    def get_relevant_docs(self) -> (dict):
        topics = self.model.get_topics()
        for i in range(0, len(topics)):
            for j in range(0, len(topics[i])):
                if topics[i][j][0] == self.keyword:
                    topic_no = i
        try:
            topic_no
        except:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__, "No topics found")
            return IXO_RET_NOT_IMPLEMENTED
        doc_id = self.model.get_top_docs(topic=topic_no, n_docs=self.top_n, sort_by='log_prob')
        doc_id = [i[0] for i in doc_id]

        lst_Exact, lst_Relevant, topic_lst = _get_documents_list(doc_id, self.docs_list, topics, self.keyword)
        doc_dict = _get_documents_dict(lst_Exact, lst_Relevant, topic_lst, self.docs_list)
        for i in doc_dict:
            lst = doc_dict[i][1]
            doc_dict[i] = list(doc_dict[i])
            doc_dict[i] = (doc_dict[i][0], _get_topic_prob_list([lst]))
        return doc_dict


    @method_header(
        description='''
                    Find keywords, which is similar to the keyword.''',
        returns='''
                    a list which contains keywords which is similar to the given keyword.''')
    def get_relevant_keywords(self) -> (list):
        topics = self.model.get_topics()
        for i in range(0, len(topics)):
            for j in range(0, len(topics[i])):
                if topics[i][j][0] == self.keyword:
                    topic_no = i
        topics = _get_topic_prob_dict(topics)
        keyword_lst = topics[topic_no]
        return keyword_lst


def _get_topic_prob_list(topics) -> (list):
    for k in topics:
        prob_tot = 0
        topic_sublist = []
        for i in range(len(k)):
            prob_tot += k[i][1]
        for i in range(len(k)):
            Second_item = round((k[i][1] / prob_tot) * 100, 2)
            topic_sublist.append((k[i][0], Second_item))
    return topic_sublist


def _get_topic_prob_dict(topics) -> (dict):
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
        topic_dict[count] = {'Corr': corr_lst, 'Inv_Corr': inv_corr_lst}
        count += 1
    return topic_dict


def _get_documents_list(doc_id, docs_list, topics, keyword) -> (list, list, list):
    topic_lst = []
    for n, topic in enumerate(topics):
        for i in topic:
            topic_lst.append(i)
    keyword_list = list(keyword.split(" "))
    lst_Exact = []
    lst_Relevant = []
    for i in doc_id:
        for keyword in keyword_list:
            if keyword in docs_list[i].lower():
                lst_Exact.append(i)
            else:
                lst_Relevant.append(i)
    lst_Exact = list(dict.fromkeys(lst_Exact))
    lst_Relevant = list(dict.fromkeys(lst_Relevant))
    lst_Relevant = set(lst_Relevant) - set(lst_Exact)
    lst_Relevant = list(lst_Relevant)
    return lst_Exact, lst_Relevant, topic_lst


def _get_documents_dict(lst_Exact, lst_Relevant, topic_lst, docs_list) -> (dict):
    doc_dict = {}
    for i in lst_Exact:
        doc_keyword_lst = []
        for j in topic_lst:
            if j[0] in docs_list[i]:
                doc_keyword_lst.append((j[1], j[0]))
        doc_keyword_lst.sort(reverse=True)
        doc_keyword_lst = [(doc_keyword_lst[k][1], doc_keyword_lst[k][0]) for k in range(len(doc_keyword_lst))]
        doc_dict[i] = docs_list[i],doc_keyword_lst
    for i in lst_Relevant:
        doc_keyword_lst = []
        for j in topic_lst:
            if j[0] in docs_list[i]:
                doc_keyword_lst.append((j[1], j[0]))
        doc_keyword_lst.sort(reverse=True)
        doc_keyword_lst = [(doc_keyword_lst[k][1], doc_keyword_lst[k][0]) for k in range(len(doc_keyword_lst))]
        doc_dict[i] = docs_list[i],doc_keyword_lst
    return doc_dict


