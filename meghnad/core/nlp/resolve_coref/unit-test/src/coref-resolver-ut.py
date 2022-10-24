from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.nlp.resolve_coref.src.coref_resolver import ResolveCoref

import os, gc
import pandas as pd
import ast

import unittest

def _cleanup():
    gc.collect()

def _write_results_tc(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    with open(results_path + "tc_resolve_coref.txt", 'w') as f:
        for profile in result:
            f.write("\n\nSequence:\n")
            f.write(str(profile['Sequence']))
            f.write("\nSequence_resolved:\n")
            f.write(str(profile['Sequence_resolved']))

def _tc_1(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "tc_1.csv")

    result = []

    for seq in tc['seq']:
        profile = {}

        profile['Sequence'] = seq
        profile['Sequence_resolved'] = model.get_coref_resolved(seq)

        result.append(profile)

    results_path += "tc_1/"
    _write_results_tc(result, results_path)

def _tc_2(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "tc_2.csv")

    result = []

    for i,row in tc.iterrows():
        profile = {}

        profile['Sequence'] = row['seq']
        profile['Sequence_resolved'] = model.get_coref_resolved(row['seq'],ast.literal_eval(row["known_tags"]))

        result.append(profile)

    results_path += "tc_2/"
    _write_results_tc(result, results_path)

def _perform_tests():
    model = ResolveCoref()

    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/nlp/resolve_coref/unit-test/"
    testcases_path = ut_path + "testcases/"
    results_path = ut_path + "results/"

    _tc_1(model, testcases_path, results_path)
    _tc_2(model, testcases_path, results_path)

if __name__ == '__main__':
    _perform_tests()

    unittest.main()

    _cleanup()

