from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.lang_translation.src.lang_translator import LangTranslator

import os, gc, json
import pandas as pd
import unittest


def _cleanup():
    gc.collect()

def _write_results_tc_1(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    
    with open(results_path + "tc_jp_results.txt", 'w', encoding = "utf8") as f:
        f.write("translated_text : \n")
        f.write(result["text"])

def _tc_1(model, testcases_path, results_path):
    result = {'tc': []}
    with open(testcases_path + "jp_text.txt", 'r', encoding = "utf8") as f:
        testcases = f.readlines()

        translated_text = model.translator(testcases)
        result = {"text":translated_text}

    results_path += "tc_jp/"
    _write_results_tc_1(result, results_path)

def _perform_tests():
    model = LangTranslator()

    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/nlp/lang_translation/unit-test/"
    testcases_path = ut_path + "testcases/"
    results_path = ut_path + "results/"

    _tc_1(model, testcases_path, results_path)

if __name__ == '__main__':
    _perform_tests()
    unittest.main()
    _cleanup()

