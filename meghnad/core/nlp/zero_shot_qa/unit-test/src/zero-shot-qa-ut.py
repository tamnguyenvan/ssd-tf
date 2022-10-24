from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.zero_shot_qa.src.zero_shot_qa import ZeroShotQA

import os, gc, json

import unittest

def _cleanup():
    gc.collect()

def _write_results_tc_1(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(results_path + "tc_results.json", 'w') as f:
        json.dump(result, f)

def _tc_1(model, testcases_path, results_path):
    result = {'tc': []}
    with open(testcases_path + "tc_1.json", 'r') as f:
        testcases = json.loads(f.read())['tc']
        for tc in testcases:
            response = model.pred(tc['seq'], tc['question'])

            result_dict = {}
            result_dict['tc_id'] = tc['tc_id']
            result_dict['seq'] = tc['seq']
            result_dict['question'] = tc['question']
            result_dict['response'] = response

            result['tc'].append(result_dict)

    results_path += "tc_1/"
    _write_results_tc_1(result, results_path)

def _write_results_tc_2(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(results_path + "tc_results.json", 'w') as f:
        json.dump(str(result), f)

def _tc_2(model, testcases_path, results_path):
    result = {'tc': []}
    with open(testcases_path + "tc_2.json", 'r') as f:
        testcases = json.loads(f.read())['tc']
        for tc in testcases:
            response = model.pred(tc['seq'], tc['sent'], top_n=2)

            result_dict = {}
            result_dict['tc_id'] = tc['tc_id']
            result_dict['seq'] = tc['seq']
            result_dict['sent'] = tc['sent']
            result_dict['response'] = response

            result['tc'].append(result_dict)

    results_path += "tc_2/"
    _write_results_tc_2(result, results_path)

def _perform_tests():
    model = ZeroShotQA()

    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/nlp/zero_shot_qa/unit-test/"
    testcases_path = ut_path + "testcases/"
    results_path = ut_path + "results/"

    _tc_1(model, testcases_path, results_path)

    semantic_model = ZeroShotQA(mode='semantic_search')

    _tc_2(semantic_model, testcases_path, results_path)

if __name__ == '__main__':
    _perform_tests()

    unittest.main()

    _cleanup()

