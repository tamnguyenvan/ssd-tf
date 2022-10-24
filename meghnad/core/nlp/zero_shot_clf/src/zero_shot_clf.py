#######################################################################################################################
# Zero-shot classifier for natural language.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from utils.common_text_preproc import strip_punctuations, strip_stop_words, get_subsuming_strings
from connectors.interfaces.interface import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.zero_shot_clf.cfg.config import ZeroShotClfConfig
from meghnad.core.nlp.detect_lang.src.detect_lang import DetectLang
from meghnad.core.nlp.text_profiler.src.text_profiler import TextProfiler

import sys, nltk, itertools, operator
from collections import OrderedDict
import numpy as np

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

log = Log()

@class_header(
description='''
Zero-shot multi-language classifier for NLP.''')
class ZeroShotClf():
    def __init__(self, *args, **kwargs):
        self.lang = None
        self.model = None
        self.hypothesis_template = None
        self.auto_detect_lang = True

        if 'hypothesis_template' in kwargs:
            hypothesis_template = kwargs['hypothesis_template']
        else:
            hypothesis_template = None

        lang = None
        if 'lang' in kwargs:
            lang = kwargs['lang']
            if lang != 'en':
                lang = 'multi'

        if 'mode' in kwargs:
            self.configs = ZeroShotClfConfig(MeghnadConfig(), mode=kwargs['mode'])
        else:
            self.configs = ZeroShotClfConfig(MeghnadConfig())

        self.detect_lang = DetectLang()
        self.text_profiler = TextProfiler(include_repeated_phrases=True,
                                          model=self.configs.get_zsc_model('en'))

        if lang:
            self._load_model(lang, hypothesis_template)
            self.auto_detect_lang = False

    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}

    @method_header(
    description='''
    Get list of languages supported.''')
    def get_langs_supported(self) -> [str]:
        return self.configs.get_langs_supported()

    @method_header(
    description='''
    Set language selected.''',
    arguments='''
    lang [optional]: language to be set
    hypothesis_template [optional]: hypothesis template to be used''',
    returns='''
    selected language''')
    def set_lang(self, lang:str=None, hypothesis_template:str=None) -> str:
        if lang:
            if lang != 'en':
                lang = 'multi'

            if self.lang != lang:
                self._load_model(lang, hypothesis_template)
            self.auto_detect_lang = False
        else:
            self.lang = None
            self.model = None
            self.hypothesis_template = None
            self.auto_detect_lang = True
            
        return self.lang

    @method_header(
        description='''
        Predict label among candidate labels provided.''',
        arguments='''
        sequence: text input for which the label needs to be predicted
        candidate_labels: candidate labels, either as a string separated by ';;' or as a list of strings
        multi_label [optional]: indicates whether multiple labels can be possible simultaneously
        explain [optional]: indicates whether attributions for top predictions need to be returned''',
        returns='''
        a 4 member tuple containing predicted label (list of labels if multi_label is True), 
        corresponding scores predicted, language detected, and attributions for top predictions (if requested) respectively''')
    def pred(self, sequence:str, candidate_labels:object,
             multi_label:bool=False, explain:bool=False) -> ([str], dict, str, dict):
        sequence = str(sequence)
        candidate_labels = _process_candidate_labels(self.configs, candidate_labels)

        if self.auto_detect_lang:
            lang = self.detect_lang.pred(sequence)
            candidate_labels_lang = self.detect_lang.pred(' '.join(candidate_labels))
            if lang != candidate_labels_lang or lang != 'en':
                lang = 'multi'
            if self.lang != lang:
                self._load_model(lang)

        if not self.model:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "State machine corrupted. Resetting the state machine and defaulting to English")
            self._load_model('en')

        response = self.model(sequence, candidate_labels, multi_label=multi_label)
        labels_pred = response['labels']
        scores_pred = response['scores']
        scores_pred = {idx: float(scores_pred.pop(0)) for idx in labels_pred}

        if not multi_label:
            labels_pred = max(scores_pred, key=lambda x: scores_pred[x])

        log.STATUS(sys._getframe().f_lineno,
                   __file__, __name__,
                   "Sequence: {}, Language: {}, Scores: {}".\
                   format(sequence, self.lang, scores_pred))

        attributions = {}
        if explain:
            if multi_label:
                min_conf = max(self.configs.get_zsc_settings('min_lift_over_rand') + 1 / len(candidate_labels),
                               scores_pred[labels_pred[0]] * self.configs.get_zsc_settings('min_pct_of_topmost_conf'))

                for label in scores_pred:
                    if scores_pred[label] >= min_conf:
                        attributions[label] = self.label_explain(sequence, label)
                    else:
                        attributions[label] = []
            else:
                label = labels_pred[0]
                attributions[label] = self.label_explain(sequence, label)

        return labels_pred, scores_pred, self.lang, attributions

    # Load appropriate pipeline
    def _load_model(self, lang:str, hypothesis_template:str=None):
        lang = str(lang)
        if lang not in self.get_langs_supported():
            log.WARNING(sys._getframe().f_lineno,
                        __file__, __name__,
                        "Language code {} not supported. Defaulting to 'multi' language".format(lang))
            lang = 'multi'
        self.lang = lang
        model_config = self.configs.get_zsc_model(self.lang)

        if hypothesis_template:
            self.hypothesis_template = hypothesis_template
        elif not self.hypothesis_template:
            self.hypothesis_template = model_config['hypothesis_template']

        self.model = pipeline('zero-shot-classification',
                              model=model_config['ckpt'],
                              hypothesis_template=self.hypothesis_template)

        self.model_expl = None
        if 'expl_ckpt' in model_config:
            self.model_expl = SentenceTransformer(model_config['expl_ckpt'])

        self.stop_words = []
        if self.lang == 'en':
            self.stop_words = nltk.corpus.stopwords.words('english')

    @method_header(
        description='''
            Get attributions for a given predicted label.''',
        arguments='''
            sequence: text input for which the label was predicted
            label: predicted label
            top_n [optional]: top n attributions to be returned
            ignore_words [optional]: list of words to be ignored''',
        returns='''
            a list of 2 member tuples containing top n words / phrases / sentences and corresponding confidences''')
    def label_explain(self, sequence:str, label:str,
                      top_n:int=None, ignore_words:[str]=[]) -> [(str, float)]:
        result = {}

        if not self.model_expl:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Model explainer not available for current usage")
            return []

        ignore_words += self.stop_words
        result['details'] = self._populate_attribution_details(sequence, label, ignore_words)

        self.seq_len_eff = len([word for word in strip_punctuations(strip_stop_words(sequence, ignore_words))])
        self._populate_top_n_attributions(result, top_n)

        self._extract_non_overlapping_attributions(result)

        return result['attributions']

    def _populate_attribution_details(self, sequence:str, label:str,
                                      ignore_words:[str]=[]) -> dict:
        if self.hypothesis_template:
            label = self.hypothesis_template.replace('{}', '{label}').format(label=label)
        self.label_emb = self.model_expl.encode(label)

        details = {}
        sents = nltk.tokenize.sent_tokenize(sequence)

        for sent_idx, sent in enumerate(sents):
            sent_emb = self.model_expl.encode(sent)
            details[sent_idx] = [(sent, util.cos_sim(self.label_emb, sent_emb).numpy()[0][0])]

            phrases = self.text_profiler.get_key_phrases(sent, aggressive=True, seed_keywords=[label])
            for phrase in phrases:
                phrase = strip_stop_words(phrase, ignore_words)
                if phrase:
                    phrase_emb = self.model_expl.encode(phrase)
                    details[sent_idx] += [(phrase, util.cos_sim(self.label_emb, phrase_emb).numpy()[0][0])]

            words = [word for word in strip_punctuations(strip_stop_words(sent, ignore_words))]
            for word in words:
                word_emb = self.model_expl.encode(word)
                details[sent_idx] += [(word, util.cos_sim(self.label_emb, word_emb).numpy()[0][0])]

            details[sent_idx].sort(key=lambda x: x[1], reverse=True)
            details[sent_idx] = list(OrderedDict(details[sent_idx]).items())
            details[sent_idx] = [(sent_idx, elem[0], elem[1]) for elem in details[sent_idx]]

        return details

    def _populate_top_n_attributions(self, result:dict,
                                     top_n:int=None):
        result['top_n'] = []
        attribution_settings = self.configs.get_zsc_settings('attributions')

        for sent_idx in result['details']:
            result['top_n'] += result['details'][sent_idx]
        result['top_n'].sort(key=lambda x: x[2], reverse=True)

        if not top_n:
            top_n = min(attribution_settings['top_n_max_limit'],
                        int(self.seq_len_eff * attribution_settings['top_n_max_pct_of_seq_len']))

        min_conf = max(self.configs.get_zsc_settings('min_lift_over_rand') + 1 / top_n,
                       result['top_n'][0][2] * self.configs.get_zsc_settings('min_pct_of_topmost_conf'))

        result['top_n'] = [elem for elem in result['top_n'] if elem[2] >= min_conf]
        result['top_n'] = result['top_n'][:top_n]

    # Get unique / non-overlapping attributions with highest normalized confidences
    def _extract_non_overlapping_attributions(self, result:dict):
        result['top_n_norm'] = [(elem[0], elem[1], self._get_attribution_norm_score(elem[1], elem[2])) \
                                for elem in result['top_n']]

        #top_n_seqs = [elem[1] for elem in result['top_n_norm']]
        #matches = get_subsuming_strings(top_n_seqs)

        result['attributions'] = result['top_n_norm']
        result['attributions'] = [(attribution[1], attribution[2]) for attribution in result['attributions']]

        new_list = self._ngrams_groupby(result['attributions'].copy())
        final_result = self._attribution_manipulator(new_list)
        log.VERBOSE(sys._getframe().f_lineno,
                    __file__, __name__,
                    final_result)

        #result['attributions'] = final_result

    # Normalze confidences based on length
    def _get_attribution_norm_score(self, attribution:str, score:float=None):
        if not score:
            attribution_emb = self.model_expl.encode(attribution)
            score = util.cos_sim(self.label_emb, attribution_emb).numpy()[0][0]
        norm_score = min(1.0, score * (1 + (np.log(len(attribution)) / self.seq_len_eff)))

        return norm_score

    def _ngrams_groupby(self, attributions:[(str, float)]) -> [[(str, float)]]:
        """
        Returns all possible grams combined together in a different lists
        received from a sequence of phrases, as an iterator.

        Input: List of tuples.
               :param sequence: the result data which is of the form (phrase, confidence)
               :type sequence: sequence or iter
             eg:[('Chocolate yummy tangy delicious', 0.3),('Chocolate yummy tangy', 0.4), ('yummy tangy delicious', 0.5),
                 ('Chocolate yummy', 0.5), ('yummy tangy', 0.45), ('tangy delicious', 0.6), ('Chocolate', 0.6),
                 ('yummy', 0.7), ('tangy', 0.56), ('delicious', 0.65)]

        Output: List of lists/tuples of strings/phrases of ngrams.
             eg: [[('Chocolate yummy tangy delicious', 0.3)],
                  [('Chocolate yummy tangy', 0.4), ('yummy tangy delicious', 0.5)],
                  [('Chocolate yummy', 0.5), ('yummy tangy', 0.45), ('tangy delicious', 0.6)],
                  [('Chocolate', 0.6), ('yummy', 0.7), ('tangy', 0.56), ('delicious', 0.65)]] """
        new_list_ngrams = []

        # Convert to list of list of tuples.
        for key, group in itertools.groupby(attributions, operator.itemgetter(0)):
            new_list_ngrams.append(list(group))

        # Add identity to tuples to help group them to a list.
        for i in range(0, len(new_list_ngrams)):
            new_list_ngrams[i].append(len(new_list_ngrams[i][0][0].split()))

        # Group ngrams of result sequence into same grams together.
        order = []
        dic = dict()

        for value, key in new_list_ngrams:
            try:
                dic[key].append(value)
            except KeyError:
                order.append(key)
                dic[key] = [value]

        return list(map(dic.get, order))

    def _attribution_manipulator(self, new_list_ngrams:[[(str, float)]]) -> [(str, float)]:
        """
        Processes result list of the zeroshot classifier to eleminate the duplicates.
        Input: List list of tuples.
        Output: List of tuples of strings/phrases of ngrams.
        """
        m = len(new_list_ngrams) - 1
        k = m - 1

        new_ls_a = []  # if not empty this will replace the child list.
        final_result_list = []  # a global result list stores the result (tuples, score).

        while m > 0:
            print("m", m)
            print("k", k)

            a = [l for l in new_list_ngrams[k]]  # Parent list changes during every iteration.
            b = [l for l in new_list_ngrams[m]]  # Child list changes during every iteration.

            print("Parent", a)
            print("child ", b)

            if not new_ls_a:
                print("new_list is empty")

                new_ls_a, result_list = self._operations_bw_two_level_lists(a, b)
                print("new_ls_a is : ", new_ls_a)

                final_result_list.append(result_list)
                print("\n")
            elif new_ls_a:
                print("since child list is not empty, new_list will replace actual child level list")
                print("new_ls_a is : ", new_ls_a)

                # new_ls_a = [tuple for list in new_ls_a for tuple in list]
                b = new_ls_a
                b = [('chocolate', 0.6), ('tangy', 0.56)]

                new_ls_a, result_list = self._operations_bw_two_level_lists(a, b)
                print("new_ls_a is : ", new_ls_a)

                final_result_list.append(result_list)
                print("\n")

            print("***************************************************************")

            m -= 1
            k -= 1

        final_result_list = [tuple for list in final_result_list for tuple in list]
        final_result_list = list(set(tuple(x) for x in final_result_list))
        # print ("final result list after iteration : ", final_result_list)

        return final_result_list

    def _operations_bw_two_level_lists(self, ls3:[(str, float)], ls4:[(str, float)]) -> [(str, float)]:
        """ Compare bw two levels, i.e parent level and child level
            1. Order the phrases to get the phrase with highest confidence value.
            2. Fetch the phrase with highest confi value from the lower order, and
               performe the operations to remove the phrase from the higher order phrase to get the new phrase.
            3. Find the confidence for the new phrase during iterative process.
            *** This function is available readily, to be used to find the confidence value for the new phrase.

        Parameters:
        Input: ls3 --> a list of tuples of Parent level.
           eg: ls3 = [('tangy delicious', 0.6), ('Chocolate yummy', 0.5), ('yummy tangy', 0.45)]

               ls4 --> a list of tuples of Child level.
           eg: ls4 = [('yummy', 0.7), ('delicious', 0.65), ('Chocolate', 0.6), ('tangy', 0.56)]

        Output:
               new_ls3 --> list of strings, replacement for next iterations' child list.
               result_list --> list of strings, will store tuples of higher confi --> [('yummy', 0.7), ('delicious', 0.65)]

        """
        ls3 = (
        sorted(ls3, key=lambda x: x[1], reverse=True))  # same as initial list of higher level phrases and confidence.
        ls4 = (sorted(ls4, key=lambda x: x[1],
                      reverse=True))  # same as final list of higher confidence phrases selected from lower level.

        print("\n----------Extraction of Phrases started:--------\n")
        print("Parent level: {}".format(ls3))
        print(" Child level: {}".format(ls4))

        new_ls3 = []  # This list stores extracted text, subtracted bw Parent & Child.
        result_list = []  # This list stores the result strings.

        for i in range(len(ls3)):
            print("\nExtract Phase: Iteration For the Parent Tuple: ", ls3[i])

            for j in range(len(ls4)):
                print("\nExtract Phase: Iteration For the Child Tuple: ", ls4[j])

                if set(ls4[j][0]).issubset(ls3[i][0]):
                    print(
                        "---------------Yes, Child -> {} is a subset of Parent tuple {}.".format(ls4[j][0], ls3[i][0]))

                    print(
                        "---------------check if Child tuples' confidence score is greater than Parent level confi score ")
                    if (ls4[j][1]) >= (ls3[i][1]):
                        print(
                            "---------------YES, Child tuple is greater than/equal to Parent level tuple, good for extraction")

                        if (ls4[j][0]) != (ls3[i][0]):
                            print("---------------Checking, if Parent tuple is same as child tuple")
                            print(
                                "---------------Child (Sub-Tuple) is not same as Parent (Top level) tuple, extraction starts now.")

                            result_list.append(ls4[j])  # append the tuples with higher confi score to result list.
                            print("---------------YES, Child tuple {} is captured in the result list.".format(ls4[j]))
                            print("---------------Now, the result list is : ", result_list)

                            parent_text = list(ls3[i][0].split())
                            print("---------------parent tuple", parent_text)

                            child_text = list(ls4[j][0].split())
                            print("---------------child tuple", child_text)

                            substracted_text = list(
                                token for token in ls3[i][0].split() if token not in ls4[j][0].split())

                            print("---------------The extracted text is {}.".format(substracted_text))

                            '''
                                ## call score function here.*************************************
                                eg: 
                                I/P: ['Chocolate']
                                O/P: ('Chocolate', 0.6)

                            '''
                            op_score = ('tangy', self._get_attribution_norm_score('tangy'))  # O/P of score function
                            print("assumed score_output : ", op_score)

                            op_score = list([op_score])

                            for t in range(len(op_score)):
                                for j in range(len(ls4)):
                                    if set(ls4[j][0]).issubset(op_score[t][0]):
                                        if (ls4[j][1]) >= (op_score[t][1]):
                                            if (ls4[j][0]) == (op_score[t][0]):
                                                print(
                                                    "---------------YES, Parent and Child tuples are same, so appending Parent to new_list")
                                                new_ls3.append(
                                                    op_score)  # list to preserve score function o/p for further iteration. ** Correct this **
                                                print("new_ls3 at this stage is : ", new_ls3)

                            break
                        else:
                            print(
                                "NO, Child tuple -> {} is same as Parent tuple -> {}, so tuples NOT extracted. ".format(
                                    ls4[j][0], ls3[i][0]))
                            break
                    else:
                        print(
                            "---------------NO, Child level tuple -> '{}' is NOT Greater than Parent level tuple -> '{}' ".format(
                                ls4[j][0], ls3[i][0]))
                        print(
                            "---------------Extraction can not be done, since Parent level tuple's confi score is greater than Child level")
                        new_ls3.append(list(ls3[i][0].split()))
                else:
                    # print ("NO, Child tuple -> {} is NOT a subset of Parent level tuple -> {}".format(ls4[j][0] , ls3[i][0])
                    pass

        new_ls3 = [tuple for list in new_ls3 for tuple in list]
        return new_ls3, result_list

# Process candidate labels into a list of strings in case it was provided as a single string separated by ';;'
def _process_candidate_labels(configs:object, candidate_labels:object, ) -> [str]:
    if isinstance(candidate_labels, list):
        candidate_labels = [str(label) for label in candidate_labels]
    elif isinstance(candidate_labels, str):
        candidate_labels = str(candidate_labels).split(configs.get_zsc_settings('multi_val_sep'))
    else:
        log.WARNING(sys._getframe().f_lineno,
                    __file__, __name__,
                    "Candidate labels provided are in wrong format")
        candidate_labels = ["Unknown"]
    return candidate_labels

if __name__ == '__main__':
    pass

