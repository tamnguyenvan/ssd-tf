from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.text_profiler.src.text_profiler import TextProfiler

import os, gc
import pandas as pd

import unittest

def _cleanup():
    gc.collect()

def _write_results_tc_1(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    with open(results_path + "tc_text_profile.txt", 'w') as f:
        for profile in result:
            f.write("\n\nSequence:\n")
            f.write(str(profile['seq']))
            f.write("\nKey phrases:\n")
            f.write(str(profile['key_phrases']))
            f.write("\nLexical features:\n")
            f.write(str(profile['lexical_features']))
            f.write("\nStylometric features:\n")
            f.write(str(profile['stylometric_features']))

def _tc_1(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "tc_1.csv")

    result = []

    for seq in tc['seq']:
        profile = {}

        profile['seq'] = seq
        profile['key_phrases'] = model.get_key_phrases(seq, aggressive=False)
        profile['lexical_features'] = model.get_lexical_features(seq)
        profile['stylometric_features'] = model.get_stylometric_features(seq)

        result.append(profile)

    results_path += "tc_1/"
    _write_results_tc_1(result, results_path)

def _perform_tests():
    model = TextProfiler()

    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/nlp/text_profiler/unit-test/"
    testcases_path = ut_path + "testcases/"
    results_path = ut_path + "results/"

    _tc_1(model, testcases_path, results_path)

if __name__ == '__main__':
    _perform_tests()

    unittest.main()

    _cleanup()

