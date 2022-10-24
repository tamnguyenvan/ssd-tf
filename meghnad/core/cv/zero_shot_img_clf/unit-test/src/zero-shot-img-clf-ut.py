from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.cv.zero_shot_img_clf.src.zero_shot_img_clf import ZeroShotImageClassification

import os, gc, json
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
import unittest

def _cleanup():
    gc.collect()


def _write_results_tc_2(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(results_path + "tc_results.json", 'w') as f:
        json.dump(result, f)

    result = pd.DataFrame(result['tc'], columns=['tc_id', 'seq', 'label_actual', 'label_pred', 'correct'])
    with open(results_path + "tc_accuracy_report.txt", 'w') as f:
        f.write("confusion_matrix:\n")
        f.write(str(confusion_matrix(result['label_actual'], result['label_pred'])))
        f.write("\n\n")
        f.write("classification_report:\n")
        f.write(str(classification_report(result['label_actual'], result['label_pred'])))

    results_incorrect_path = results_path + "result_incorrect.csv"
    result_incorrect = result[result['correct'] == False]
    if result_incorrect.size:
        result_incorrect.to_csv(results_incorrect_path)
    elif os.path.exists(results_incorrect_path):
        os.remove(results_incorrect_path)

def _tc_2(model, testcases_path, results_path):
    result = {'tc': []}
    with open(testcases_path + "tc_2.json", 'r') as f:
        testcases = json.loads(f.read())['tc']
        for tc in testcases:
            label_pred, scores_pred, lang = model.pred(tc['seq'], tc['candidate_labels'])

            result_dict = {}
            result_dict['tc_id'] = tc['tc_id']
            result_dict['seq'] = tc['seq']
            result_dict['label_actual'] = tc['label_actual']
            result_dict['lang'] = lang
            result_dict['scores_pred'] = str(scores_pred[0])
            result_dict['label_pred'] = label_pred[0]
            result_dict['correct'] = result_dict['label_actual'] == result_dict['label_pred']

            result['tc'].append(result_dict)

    results_path += "tc_2/"
    _write_results_tc_2(result, results_path)

def _write_results_tc_1(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(results_path + "tc_results.json", 'w') as f:
        json.dump(result, f)

    result = pd.DataFrame(result['tc'], columns=['tc_id', 'seq', 'label_actual', 'label_pred', 'correct'])
    with open(results_path + "tc_accuracy_report.txt", 'w') as f:
        f.write("confusion_matrix:\n")
        f.write(str(confusion_matrix(result['label_actual'], result['label_pred'])))
        f.write("\n\n")
        f.write("classification_report:\n")
        f.write(str(classification_report(result['label_actual'], result['label_pred'])))

    results_incorrect_path = results_path + "result_incorrect.csv"
    result_incorrect = result[result['correct'] == False]
    if result_incorrect.size:
        result_incorrect.to_csv(results_incorrect_path)
    elif os.path.exists(results_incorrect_path):
        os.remove(results_incorrect_path)

def _tc_1(model, testcases_path, results_path):
    result = {'tc': []}
    with open(testcases_path + "tc_1.json", 'r') as f:
        testcases = json.loads(f.read())['tc']
        for tc in testcases:
            label_pred, scores_pred, lang = model.pred(tc['seq'], tc['candidate_labels'])

            result_dict = {}
            result_dict['tc_id'] = tc['tc_id']
            result_dict['seq'] = tc['seq']
            result_dict['label_actual'] = tc['label_actual']
            result_dict['lang'] = lang
            result_dict['scores_pred'] = str(scores_pred[0])
            result_dict['label_pred'] = label_pred[0]
            result_dict['correct'] = result_dict['label_actual'] == result_dict['label_pred']

            result['tc'].append(result_dict)
    
    results_path += "tc_1/"
    _write_results_tc_1(result, results_path)

def _perform_tests():
    model = ZeroShotImageClassification()

    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/cv/zero_shot_img_clf/unit-test/"
    testcases_path = ut_path + "testcases/"
    results_path = ut_path + "results/"

    _tc_1(model, testcases_path, results_path)

    _tc_2(model, testcases_path, results_path)

if __name__ == '__main__':
    _perform_tests()
    
    unittest.main()

    _cleanup()

