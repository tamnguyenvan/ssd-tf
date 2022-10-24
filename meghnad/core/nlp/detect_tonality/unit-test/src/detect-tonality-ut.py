from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.metrics.metrics import ClfMetrics
from meghnad.core.nlp.detect_tonality.src.detect_tonality import DetectTonality

import os, gc
from ast import literal_eval
import pandas as pd

import unittest

def _cleanup():
    gc.collect()

def _write_results_tc_1(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    result.to_csv(results_path + "tc_results.csv")

    clf_metrics = ClfMetrics()
    cnf_mat, clf_rep = clf_metrics.default_metrics(result['tonality'], result['labels_pred'], multi_label=True)
    with open(results_path + "tc_accuracy_report.txt", 'w') as f:
        f.write("confusion_matrix:\n")
        f.write(str(cnf_mat))
        f.write("\n\n")
        f.write("classification_report:\n")
        f.write(str(clf_rep))

    results_incorrect_path = results_path + "result_incorrect.csv"
    result_incorrect = result[result['correct'] == False]
    if result_incorrect.size:
        result_incorrect['tonality'] = result_incorrect['tonality'].astype(str)
        result_incorrect['labels_pred'] = result_incorrect['labels_pred'].astype(str)
        result_incorrect['scores_pred'] = result_incorrect['scores_pred'].astype(str)

        result_incorrect.to_csv(results_incorrect_path)
    elif os.path.exists(results_incorrect_path):
        os.remove(results_incorrect_path)

def _tc_1(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "tc_1.csv")

    result = tc
    labels_pred = []
    scores_pred = []

    for seq in result['seq']:
        label_pred, score_pred = model.pred(seq)
        assert(isinstance(label_pred, list) and isinstance(score_pred, list))
        scores_pred.append(score_pred)
        labels_pred.append(label_pred)

    result['labels_pred'] = labels_pred
    result['scores_pred'] = scores_pred

    if isinstance(result['tonality'][0], str):
        result['tonality'] = result['tonality'].apply(lambda x: literal_eval(str(x)))

    result['correct'] = result.apply(lambda row: set(row['tonality']) == set(row['labels_pred']), axis=1)

    results_path += "tc_1/"
    _write_results_tc_1(result, results_path)

def _perform_tests():
    model = DetectTonality()

    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/nlp/detect_tonality/unit-test/"
    testcases_path = ut_path + "testcases/"
    results_path = ut_path + "results/"

    _tc_1(model, testcases_path, results_path)

if __name__ == '__main__':
    _perform_tests()

    unittest.main()

    _cleanup()

