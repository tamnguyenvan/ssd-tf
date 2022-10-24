from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.detect_lang.src.detect_lang import DetectLang

import os, gc
import pandas as pd

import unittest

from sklearn.metrics import confusion_matrix, classification_report

def _cleanup():
    gc.collect()

def _write_results_tc_1(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    result.to_csv(results_path + "tc_results.csv")

    with open(results_path + "tc_accuracy_report.txt", 'w') as f:
        f.write("confusion_matrix:\n")
        f.write(str(confusion_matrix(result['lang'], result['lang_pred'])))
        f.write("\n\n")
        f.write("classification_report:\n")
        f.write(str(classification_report(result['lang'], result['lang_pred'])))

    results_incorrect_path = results_path + "result_incorrect.csv"
    result_incorrect = result[result['correct'] == False]
    if result_incorrect.size:
        result_incorrect.to_csv(results_incorrect_path)
    elif os.path.exists(results_incorrect_path):
        os.remove(results_incorrect_path)

def _tc_1(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "tc_1.csv")

    result = tc
    result['lang_pred'] = [model.pred(seq) for seq in result['seq']]
    result['correct'] = result['lang'].equals(result['lang_pred'])

    results_path += "tc_1/"
    _write_results_tc_1(result, results_path)

def _write_results_tc_2(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    with open(results_path + "tc_method_doc.txt", 'w') as f:
        f.write("Method doc:\n")
        f.write(str(result))

    with open(results_path + "tc_class_doc.txt", 'w') as f:
        f.write("Class dict doc:")
        f.write(DetectLang.__dict__['__doc__'])
        f.write("\nClass dict:\n")
        f.write(str(DetectLang.__dict__))
        f.write("Class dir:\n")
        f.write(str(dir(DetectLang)))

def _tc_2(model, results_path):
    result = model.pred.__doc__

    results_path += "tc_2/"
    _write_results_tc_2(result, results_path)

def _perform_tests():
    model = DetectLang()

    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/nlp/detect_lang/unit-test/"
    testcases_path = ut_path + "testcases/"
    results_path = ut_path + "results/"

    _tc_1(model, testcases_path, results_path)

    _tc_2(model, results_path)

if __name__ == '__main__':
    _perform_tests()

    unittest.main()

    _cleanup()

