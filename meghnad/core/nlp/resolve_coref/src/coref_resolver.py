#######################################################################################################################
# Resolve coreference in English text.
#
#  Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Nayan Sarkar
#######################################################################################################################

from xmlrpc.client import Boolean
from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from connectors.interfaces.interface import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.resolve_coref.cfg.config import ResolveCorefConfig

import sys, spacy, coreferee
from typing import List
from spacy.tokens import Doc, Span
from fuzzywuzzy import fuzz
from allennlp.predictors.predictor import Predictor

log = Log()

@class_header(
description='''
Coreference Resolution for NLP.''')
class ResolveCoref():
    def __init__(self, *args, **kwargs):
        self.configs = ResolveCorefConfig(MeghnadConfig())

        self.predictor = Predictor.from_path(self.configs.get_resolve_coref_cfg().get('allenNLPurl'))
        self.coref_nlp = spacy.load('en_core_web_trf')
        self.coref_nlp.add_pipe('coreferee')

        self.nlp = spacy.load('en_core_web_lg')

    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}

    @method_header(
        description='''
        Get coreference resolved.''',
        arguments='''
        seq: text input''',
        returns='''
        dictionary of sentences and tags''')
    def get_coref_resolved(self, seq:str, known_tags:[str]=None, *args, **kwargs) -> dict:
        coref_dict = dict()

        clusters = self.predictor.predict(seq)['clusters']
        doc = self.coref_nlp(seq)
        log.VERBOSE(sys._getframe().f_lineno,
                    __file__, __name__,
                    "Sequence parsing complete")

        improved_text = self._improved_replace_corefs(doc,clusters)

        log.VERBOSE(sys._getframe().f_lineno,
                    __file__, __name__,
                    "Coreference Resolved for the sequence")

        improved_doc = self.nlp(improved_text)
        for i,sent in enumerate(improved_doc.sents):
            proper_nouns = [token.text for token in sent if token.pos_ == "PROPN"]
            ncs = []
            for nc in sent.noun_chunks:
                
                small_doc = self.nlp(nc.text)
                if len([token.pos_ for token in small_doc if token.pos_ in ["PROPN"] or token.text in proper_nouns]) > 0:
                    if known_tags == None:
                        ncs.append(str(nc))
                        log.VERBOSE(sys._getframe().f_lineno,
                                    __file__, __name__,
                                    "No external tags provided")
                    elif known_tags != None:
                        matches = self._get_matches(nc.text,known_tags)
                        if len(matches) > 0:
                            for match in matches:
                                ncs.append(match[1])
                                try:
                                    sent = sent.text
                                    sent = sent.replace(nc.text,matches[0][1])
                                except:
                                    sent = sent.replace(nc.text,matches[0][1])

            coref_dict[i] =\
            {   "seq":sent,
                "tags":list(set(ncs))
            }


        return coref_dict

    def _word_match(self, word_1:str,word_2:str) -> Boolean:
        if fuzz.ratio(word_1,word_2) > 70:
            return True
        else:
            return False

    def _get_matches(self, string_1:str, known_tags:List) -> List:
        matched_strings = []
        for tag in known_tags:
            for i in string_1.split():
                if self._word_match(i,tag):
                    matched_strings.append((string_1,tag))
                    break
        
        return matched_strings

    def _core_logic_part(self,document: Doc, coref: List[int], resolved: List[str], mention_span: Span):
        final_token = document[coref[1]]
        if final_token.tag_ in self.configs.get_resolve_coref_cfg().get('tag'):
            resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
        else:
            resolved[coref[0]] = mention_span.text + final_token.whitespace_
        for i in range(coref[0] + 1, coref[1] + 1):
            resolved[i] = ""
        return resolved

    def _get_span_noun_indices(self,doc: Doc, cluster: List[List[int]]) -> List[int]:
        spans = [doc[span[0]:span[1]+1] for span in cluster]
        spans_pos = [[token.pos_ for token in span] for span in spans]
        span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
            if any(pos in span_pos for pos in self.configs.get_resolve_coref_cfg().get('pos'))]
        return span_noun_indices

    def _get_cluster_head(self,doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
        head_idx = noun_indices[0]
        head_start, head_end = cluster[head_idx]
        head_span = doc[head_start:head_end+1]
        return head_span, [head_start, head_end]

    def _is_containing_other_spans(self,span: List[int], all_spans: List[List[int]]):
        return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])

    def _improved_replace_corefs(self,document, clusters) -> str:
        resolved = list(tok.text_with_ws for tok in document)
        all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

        for cluster in clusters:
            noun_indices = self._get_span_noun_indices(document, cluster)

            if noun_indices:
                mention_span, mention = self._get_cluster_head(document, cluster, noun_indices)

                for coref in cluster:
                    if coref != mention and not self._is_containing_other_spans(coref, all_spans):
                        self._core_logic_part(document, coref, resolved, mention_span)

        return "".join(resolved)

if __name__ == '__main__':
    pass

