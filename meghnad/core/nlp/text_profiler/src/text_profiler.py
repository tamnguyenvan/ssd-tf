#######################################################################################################################
# Profiler for English language.
#
#  Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from connectors.interfaces.interface import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.text_profiler.cfg.config import TextProfilerConfig

import sys, math, string, nltk
import collections as coll
import scipy as sc
import numpy as np

from keybert import KeyBERT
from rake_nltk import Rake
import pke, yake
from nltk.util import ngrams

log = Log()

@class_header(
description='''
Profiler for NLP.''')
class TextProfiler():
    def __init__(self, *args, **kwargs):
        self.configs = TextProfilerConfig(MeghnadConfig())

        pke_extractor = pke.unsupervised.MultipartiteRank()

        yake_extractor = yake.KeywordExtractor(lan='en')

        if 'include_repeated_phrases' in kwargs:
            include_repeated_phrases = kwargs['include_repeated_phrases']
        else:
            include_repeated_phrases = False
        rake_extractor = Rake(include_repeated_phrases=include_repeated_phrases)

        if 'model' in kwargs:
            key_bert_extractor = KeyBERT(model=kwargs['model'])
        else:
            key_bert_extractor = KeyBERT()

        self.kp_models = [pke_extractor, yake_extractor, key_bert_extractor, rake_extractor]

    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}

    @method_header(
        description='''
        Get key phrases.''',
        arguments='''
        seq: text input
        top_n: top n key phrases to return (valid only if aggressive == False)
        aggressive [optional]: indicates whether aggressive list of key phrases to be returned
        seed_keywords [optional]: seed keywords that may guide the extraction of keywords by steering the 
        similarities towards them (applicable only if aggressive is True)''',
        returns='''
        list of of key phrases''')
    def get_key_phrases(self, seq:str, top_n:int=None,
                        aggressive:bool=False, seed_keywords:[str]=None, *args, **kwargs) -> [str]:
        if not top_n:
            top_n = int(len(seq.split()) * self.configs.get_text_profiler_cfg()['key_phrase']['top_n_as_pct_of_seq_len'])

        # Key phrase model 0
        self.kp_models[0].load_document(input=seq, language='en')
        self.kp_models[0].candidate_selection()
        self.kp_models[0].candidate_weighting()
        pke_kp = self.kp_models[0].get_n_best(n=top_n)
        pke_kp = [phrase[0] for phrase in pke_kp]
        key_phrases = pke_kp

        if aggressive:
            # Key phrase model 1
            yake_kp = self.kp_models[1].extract_keywords(seq)
            yake_kp = [phrase[0] for phrase in yake_kp]

            # Key phrase model 2
            nr_candidates = max(self.configs.get_text_profiler_cfg()['key_phrase']['nr_candidates'], top_n)
            rake_kp = self.kp_models[2].extract_keywords(seq,
                                                    top_n=top_n, nr_candidates=nr_candidates,
                                                    keyphrase_ngram_range=(1, 4),
                                                    use_maxsum=True,
                                                    stop_words=[], #'english',
                                                    seed_keywords=seed_keywords)

            rake_kp.sort(key=lambda x: x[1], reverse=True)

            min_score = self.configs.get_text_profiler_cfg()['key_phrase']['rake_min_score']
            rake_kp = [phrase for phrase in rake_kp if phrase[1] > min_score]

            rake_kp = [phrase[0] for phrase in rake_kp]

            # Key phrase model 3
            self.kp_models[3].extract_keywords_from_text(seq)
            key_bert_kp = self.kp_models[3].get_ranked_phrases_with_scores()

            min_score = self.configs.get_text_profiler_cfg()['key_phrase']['key_bert_min_score']
            key_bert_kp = [phrase for phrase in key_bert_kp if phrase[0] > min_score]

            key_bert_kp = [phrase[1] for phrase in key_bert_kp]

            # Combine
            aggresive_kp = list(set(yake_kp + rake_kp + key_bert_kp))
            key_phrases = list(set(pke_kp + aggresive_kp))

        # Handling in case words with certain select POS tags only be picked as key phrases
        pos_filter = self.configs.get_text_profiler_cfg()['key_phrase']['pos_filter']
        if pos_filter:
            for idx, phrase in enumerate(key_phrases):
                if len(phrase.split()) == 1:
                    if nltk.pos_tag(phrase)[0][1] not in pos_filter:
                        del key_phrases[idx]

        # Multiple words with same root should not be present in the key phrases returned
        lemmatizer = nltk.stem.WordNetLemmatizer()
        for idx, phrase in enumerate(key_phrases):
            if len(phrase.split()) == 1:
                root = lemmatizer.lemmatize(phrase)
                if root != phrase:
                    search_space = key_phrases.copy()
                    del search_space[idx]
                    search_space = [phrase for phrase in search_space]

                    if root in search_space:
                        del key_phrases[idx]

        return key_phrases

    @method_header(
        description='''
            Get lexical features.''',
        arguments='''
            seq: text input''',
        returns='''
            dictionary containing features''')
    def get_lexical_features(self, seq:str) -> dict:
        seq_str = self._get_seq_structure(seq)

        features = {}

        features['word_cnt'] = len(seq_str['words'])
        features['sent_cnt'] = len(seq_str['sents'])
        features['para_cnt'] = len(seq_str['paras'])

        features['char_cnt_in_longest_word'] = len(seq_str['longest_word'])
        features['word_cnt_in_longest_sent'] = len(seq_str['longest_sent'].split())
        features['word_cnt_in_longest_para'] = len(seq_str['longest_para'].split())

        features['avg_sents_per_para'] = round(features['sent_cnt'] / max(1, features['para_cnt']),2)
        features['avg_words_per_sent'] = round(len(seq_str['words_excl_stop']) / max(1, features['sent_cnt']),2)
        features['avg_chars_per_word'] = round(np.average([len(word) for word in seq_str['words_excl_stop']]),2)

        special_char_cnt = len([char for char in seq if char in self.configs.get_text_profiler_cfg()['special_chars']])
        features['avg_special_chars_per_sent'] = round(special_char_cnt / max(1, features['sent_cnt']),2)

        punct_cnt = len([char for char in seq if char in list(string.punctuation)])
        features['avg_puncts_per_sent'] = round(punct_cnt / max(1, features['sent_cnt']),2)

        functional_word_cnt = len([word for word in seq_str['words_excl_stop'] \
                                   if word in self.configs.get_text_profiler_cfg()['functional_words']])
        features['avg_functional_words_per_sent'] = round(functional_word_cnt / max(1, features['sent_cnt']),2)

        return features

    @method_header(
        description='''
                Get stylometric features.''',
        arguments='''
                seq: text input''',
        returns='''
                dictionary containing features''')
    def get_stylometric_features(self, seq:str) -> dict:
        seq_str = self._get_seq_structure(seq)

        features = {}

        features['vocab_richness'] = _get_vocab_richness_features(seq_str)
        features['readability'] = _get_readability_features(seq_str)

        return features

    def _get_seq_structure(self, seq:str):
        seq_str = {}

        seq_str['words'] = seq.split()
        seq_str['sents'] = nltk.tokenize.sent_tokenize(seq)
        
        seq_str['paras'] = [para for para in seq.splitlines()\
                            if len(para) >= self.configs.get_text_profiler_cfg()['min_chars_in_para']]

        seq_str['longest_word'] = max(seq_str['words'], key=len)
        seq_str['longest_sent'] = max(seq_str['sents'], key=len)
        seq_str['longest_para'] = max(seq_str['paras'], key=len)

        seq_punct_stripped = seq.translate(str.maketrans('', '', string.punctuation))
        seq_str['words_excl_stop'] = [word for word in seq_punct_stripped.split()\
                                      if word not in self.configs.get_text_profiler_cfg()['stop_words']]

        return seq_str

def _get_vocab_richness_features(seq_str:dict) -> dict:
    features = {}

    features['ttr'] = _type_token_ratio(seq_str)
    features['r'], features['hapax'] = _hapax_legemena(seq_str)
    features['s'], features['di_hapax'] = _hapax_dis_legemena(seq_str)
    features['k'] = _yules_characteristic(seq_str)
    features['d'] = _simpsons_index(seq_str)
    features['w'] = _brunets_measure(seq_str)
    features['h'] = _shannon_entropy(seq_str)
    features['j'] = _guiraud_r(seq_str)
    features['c'] = _herdan_c(seq_str)
    features['d'] = _dugast_k(seq_str)
    features['u'] = _dugast_u(seq_str)
    features['m'] = _maas_a(seq_str)
    features['t'] = _tuldava_score(seq_str)
    features['avg_bigrams'] = _avg_bigrams(seq_str)
    features['avg_trigrams'] = _avg_trigrams(seq_str)
    features['avg_lemma'] = _avg_lemma(seq_str)

    return features

def _get_readability_features(seq_str:dict) -> dict:
    features = {}

    syllable_cnt = sum([_syllable_cnt(word) for word in seq_str['words_excl_stop']])
    features['avg_syllables_per_word'] = round(syllable_cnt / max(1, len(seq_str['words_excl_stop'])),2)

    features['i'] = _flesch_reading_ease(seq_str)
    features['f'] = _flesch_cincade_grade_level(seq_str)
    features['g'] = _gunning_fox_index(seq_str)
    features['l'] = _coleman_liau_index(seq_str)
    features['a'] = _automated_readability_index(seq_str)

    return features

def _type_token_ratio(seq_str:dict) -> float:
    return round(len(set(seq_str['words_excl_stop'])) / max(1, len(seq_str['words_excl_stop'])),2)

def _hapax_legemena(seq_str:dict) -> (float, float):
    words = seq_str['words_excl_stop']
    V1 = 0

    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    for word in freqs:
        if freqs[word] == 1:
            V1 += 1
    N = len(words)
    V = float(len(set(words)))
    R = 100 * math.log(N) / max(1, (1 - (V1 / V)))  # Honore Measure R
    h = V1 / N

    return round(R,2), round(h,2)

def _hapax_dis_legemena(seq_str:dict) -> (float, float):
    words = seq_str['words_excl_stop']
    count = 0

    freqs = coll.Counter()
    freqs.update(words)
    for word in freqs:
        if freqs[word] == 2:
            count += 1

    h = count / float(len(words))
    S = count / float(len(set(words)))  # Sicheles Measure S

    return round(S,2), round(h,2)

def _coleman_liau_index(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    total_letters = sum([len(w) for w in words])

    L = (total_letters / len(words)) * 100
    S = (len(seq_str['sents']) / len(words)) * 100
    D = (0.0588 * L) - (0.296 * S) - 15.8
    
    return round(D,2)

def _automated_readability_index(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    total_letters = sum([len(w) for w in words])

    L = (total_letters / len(words))
    S = (len(words) / len(seq_str['sents']))
    D = (4.71 * L) + (0.5 * S) - 21.43

    return round(D,2)

def _avg_lemma(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']

    lemmatizer = nltk.stem.WordNetLemmatizer()

    lemma_list = []
    for token in words:
        lemma_list.append(lemmatizer.lemmatize(token))

    lemma_vocab = coll.Counter(lemma_list)
    avg_lemma = len(lemma_vocab.keys())/len(seq_str['sents'])

    return round(avg_lemma,2)

def _avg_bigrams(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    avg_bigrams = len(list(ngrams(words,2)))/len(seq_str['sents'])
    return round(avg_bigrams,2)

def _avg_trigrams(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    avg_trigrams = len(list(ngrams(words,3)))/len(seq_str['sents'])
    return round(avg_trigrams,2)

def _guiraud_r(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    freqs = coll.Counter()
    freqs.update(words)
    r = len(freqs.keys()) / math.sqrt(len(words))
    return round(r,2)

def _herdan_c(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    freqs = coll.Counter()
    freqs.update(words)
    c = math.log(len(freqs.keys())) / math.log(len(words))
    return round(c,2)

def _dugast_k(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    freqs = coll.Counter()
    freqs.update(words)
    k = math.log(len(freqs.keys())) / math.log(math.log(len(words)))
    return round(k,2)

def _maas_a(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    freqs = coll.Counter()
    freqs.update(words)
    a = (math.log(len(words)) - math.log(len(freqs.keys()))) / (math.log(len(words)) ** 2)
    return round(a,2)

def _dugast_u(seq_str:dict) -> float:
    # Decrease the vocabulary size by 1 if we only have hapaxes
    words = seq_str['words_excl_stop']
    freqs = coll.Counter()
    freqs.update(words)
    vocabulary_size = len(freqs.keys())
    if len(words) == vocabulary_size:
        vocabulary_size -= 1
    
    u = (math.log(len(words)) ** 2) / (math.log(len(words)) - math.log(vocabulary_size))

    return round(u,2)

def _tuldava_score(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    freqs = coll.Counter()
    freqs.update(words)
    vocabulary_size = len(freqs.keys())
    k = (1 - (vocabulary_size ** 2)) / ((vocabulary_size ** 2) * math.log(len(words)))
    return round(k,2)

def _yules_characteristic(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    N = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    vi = coll.Counter()
    vi.update(freqs.values())
    M = sum([(value * value) * vi[value] for key, value in freqs.items()])
    K = 10000 * (M - N) / math.pow(N, 2)

    return round(K,2)

def _simpsons_index(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']

    freqs = coll.Counter()
    freqs.update(words)
    N = len(words)
    n = sum([1.0 * i * (i - 1) for i in freqs.values()])
    D = 1 - (n / (N * (N - 1)))

    return round(D,2)

def _brunets_measure(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']

    a = 0.17
    V = float(len(set(words)))
    N = len(words)
    W = (V - a) / (math.log(N))

    return round(W,2)

def _shannon_entropy(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']

    lenght = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    arr = np.array(list(freqs.values()))
    distribution = 1. * arr
    distribution /= max(1, lenght)

    H = sc.stats.entropy(distribution, base=2)
    #H = sum([(i/lenght)*math.log(i/lenght,math.e) for i in freqs.values()])

    return round(H,2)

def _syllable_cnt_manual(word:str) -> int:
    vowels = 'aeiouy'

    word = word.lower()
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if not count:
        count += 1
    return count

def _syllable_cnt(word:str) -> int:
    d = nltk.corpus.cmudict.dict()

    try:
        count = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        count = _syllable_cnt_manual(word)
    return count

def _flesch_reading_ease(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    sent_cnt = len(seq_str['sents'])

    l = float(len(words))
    scount = 0
    for word in words:
        scount += _syllable_cnt(word)

    I = 206.835 - 1.015 * (l / float(sent_cnt)) - 84.6 * (scount / float(l))

    return round(I,2)

def _flesch_cincade_grade_level(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    sent_cnt = len(seq_str['sents'])

    scount = 0
    for word in words:
        scount += _syllable_cnt(word)

    l = len(words)
    F = 0.39 * (l / sent_cnt) + 11.8 * (scount / float(l)) - 15.59

    return round(F,2)

def _gunning_fox_index(seq_str:dict) -> float:
    words = seq_str['words_excl_stop']
    word_cnt = len(words)
    sent_cnt = len(seq_str['sents'])

    complex_word_cnt = 0
    for word in words:
        if (_syllable_cnt(word) > 2):
            complex_word_cnt += 1

    G = 0.4 * ((word_cnt / sent_cnt) + 100 * (complex_word_cnt / word_cnt))

    return round(G,2)

if __name__ == '__main__':
    pass