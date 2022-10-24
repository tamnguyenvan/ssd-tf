#######################################################################################################################
# Interface for NLP component in Meghnad.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.ret_values import *

from meghnad.metrics.metrics import ClfMetrics

from meghnad.core.nlp.detect_lang.src.detect_lang import DetectLang

from meghnad.core.nlp.lang_translation.src.lang_translator import LangTranslator

from meghnad.core.nlp.zero_shot_clf.src.zero_shot_clf import ZeroShotClf

from meghnad.core.nlp.detect_sentiment.src.detect_sentiment import DetectSentiment

from meghnad.core.nlp.text_clf.src.trn import TextClfTrn
from meghnad.core.nlp.text_clf.src.pred import TextClfPred

from meghnad.core.nlp.detect_tonality.src.detect_tonality import DetectTonality

from meghnad.core.nlp.detect_emotion.src.detect_emotion import DetectEmotion

from meghnad.core.nlp.text_profiler.src.text_profiler import TextProfiler

from meghnad.core.nlp.resolve_coref.src.coref_resolver import ResolveCoref

from meghnad.core.nlp.zero_shot_qa.src.zero_shot_qa import ZeroShotQA

from meghnad.core.nlp.topic_model.src.trn import TopicModelTrn
from meghnad.core.nlp.topic_model.src.pred import TopicModelPred

