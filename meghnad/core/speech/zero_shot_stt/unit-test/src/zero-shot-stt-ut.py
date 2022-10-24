from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.speech.zero_shot_stt.src.zero_shot_stt import ZeroShotSTT

import os, gc
import pandas as pd

import unittest, difflib

def _cleanup():
    gc.collect()

def _write_results(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    result.to_csv(results_path + "tc_results.csv")

def _tc_1(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "tc_1.csv")

    result = tc
    result['pred'] = [model.pred(testcases_path + file_name) for file_name in result['file_name']]

    result['diff'] = str([None])
    result['correct'] = 1

    d = difflib.Differ()
    sm = difflib.SequenceMatcher(None)

    for idx, row in result.iterrows():
        actual_words = row['actual'].lower().split()
        pred_words = row['pred'].lower().split()

        sm.set_seqs(actual_words, pred_words)
        result.at[idx, 'correct'] = sm.ratio()

        result.at[idx, 'diff'] = str([(token[2:], token[0]) if token[0] != " " else None\
                                        for token in d.compare(actual_words, pred_words)])

    results_path += "tc_1/"
    _write_results(result, results_path)

def _tc_2(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "tc_2.csv")

    result = tc
    result['pred'] = str([None])

    for idx, row in result.iterrows():
        model.convert(testcases_path + row['src_file_name'], testcases_path + row['dst_file_name'])

        result.at[idx, 'pred'] = model.pred(testcases_path + row['dst_file_name'])

        os.remove(testcases_path + row['dst_file_name'])

    results_path += "tc_2/"
    _write_results(result, results_path)

def _tc_3(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "tc_3.csv")

    result = tc
    result['pred'] = [model.pred(testcases_path + file_name) for file_name in result['file_name']]

    results_path += "tc_3/"
    _write_results(result, results_path)

def _tc_4(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "tc_4.csv")

    result = tc
    result['pred'] = [model.pred(testcases_path + row['file_name'], num_speakers=row['num_speakers'])\
                      for _, row in result.iterrows()]

    results_path += "tc_4/"
    _write_results(result, results_path)

def _perform_tests():
    model = ZeroShotSTT()

    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/speech/zero_shot_stt/unit-test/"
    testcases_path = ut_path + "testcases/"
    results_path = ut_path + "results/"

    _tc_1(model, testcases_path, results_path)

    _tc_2(model, testcases_path, results_path)

    model_diarize = ZeroShotSTT(mode='diarization')

    _tc_3(model_diarize, testcases_path, results_path)

    model_diarize = ZeroShotSTT(num_speakers_known=True, mode='diarization')

    _tc_4(model_diarize, testcases_path, results_path)

if __name__ == '__main__':
    _perform_tests()

    unittest.main()

    _cleanup()

