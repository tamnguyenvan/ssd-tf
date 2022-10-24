from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.metrics.metrics import ClfMetrics
from meghnad.core.nlp.text_clf.src.trn import TextClfTrn
from meghnad.core.nlp.text_clf.src.pred import TextClfPred

import os, gc, json
from ast import literal_eval
import pandas as pd

import unittest

def _cleanup():
    gc.collect()

def _write_results_tc_1(ret_val, saved_model_dir, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(results_path + "tc_results.txt", 'w') as f:
        f.write("ret val: ")
        f.write(str(ret_val))
        f.write("\nsaved model dir: ")
        f.write(str(saved_model_dir))

def _tc_1(model, testcases_path, results_path):
    data_path = testcases_path + "intent_trn.csv"
    data_type = 'csv'
    data_org = 'single_file'
    feature_cols = ['Utterance']
    target_cols = ['Intent']

    model.config_connectors(data_path=data_path, data_type=data_type, data_org=data_org,
                            feature_cols=feature_cols, target_cols=target_cols)

    epochs = 2
    class_balance = True

    ret_val, saved_model_dir = model.trn(epochs=epochs,
                                         class_balance=class_balance)

    results_path += "tc_1/"
    _write_results_tc_1(ret_val, saved_model_dir, results_path)

    if ret_val == IXO_RET_SUCCESS:
        return saved_model_dir

def _write_results_tc_2(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    result.to_csv(results_path + "tc_results.csv")

    clf_metrics = ClfMetrics()
    cnf_mat, clf_rep = clf_metrics.default_metrics(result['Intent'], result['labels_pred'])
    with open(results_path + "tc_accuracy_report.txt", 'w') as f:
        f.write("confusion_matrix:\n")
        f.write(str(cnf_mat))
        f.write("\n\n")
        f.write("classification_report:\n")
        f.write(str(clf_rep))

    results_incorrect_path = results_path + "result_incorrect.csv"
    result_incorrect = result[result['correct'] == False]
    if result_incorrect.size:
        result_incorrect.to_csv(results_incorrect_path)
    elif os.path.exists(results_incorrect_path):
        os.remove(results_incorrect_path)

def _tc_2(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "intent_test.csv")

    result = tc
    labels_pred = []
    scores_pred = []
    for seq in result['Utterance']:
        ret_val, label_pred, score_pred, _ = model.pred(seq)
        assert(not (isinstance(label_pred, list) or isinstance(score_pred, list)))
        if ret_val == IXO_RET_SUCCESS:
            labels_pred.append(label_pred)
            scores_pred.append(score_pred)
    result['labels_pred'] = labels_pred
    result['scores_pred'] = scores_pred
    result['correct'] = result['Intent'].equals(result['labels_pred'])

    results_path += "tc_2/"
    _write_results_tc_2(result, results_path)

def _write_results_tc_3(ret_val, saved_model_dir, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(results_path + "tc_results.txt", 'w') as f:
        f.write("ret val: ")
        f.write(str(ret_val))
        f.write("\nsaved model dir: ")
        f.write(str(saved_model_dir))

def _tc_3(model, testcases_path, results_path):
    data_path = testcases_path + "intent_trn.csv"
    data_type = 'csv'
    data_org = 'single_file'
    feature_cols = ['Utterance']
    target_cols = ['Intent']

    model.config_connectors(data_path=data_path, data_type=data_type, data_org=data_org,
                            feature_cols=feature_cols, target_cols=target_cols)

    epochs = 2
    batch_size = 8
    optimizer = ['adam', 'rmsprop']
    learning_rate = [None, 1e-3]
    decay_rate_per_10_steps = [None, 0.9, 0.8, 0.0]
    hidden_layers = [({'type': 'Dense', 'units': 64, 'activation': 'relu'}, {'type': 'Dropout', 'rate': 0.1}),
                     ({'type': 'Dense', 'units': 32, 'activation': 'tanh'}),
                     ({'type': 'Dense', 'units': 128, 'activation': 'relu'},\
                      {'type': 'Reshape', 'target_shape': (128, 1)},\
                      {'type': 'Conv1D', 'filters': 32, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'},\
                      {'type': 'GlobalMaxPool1D'})]

    ret_val, saved_model_dir = model.trn(epochs=epochs,
                                         batch_size=batch_size,
                                         optimizer=optimizer,
                                         learning_rate=learning_rate,
                                         decay_rate_per_10_steps=decay_rate_per_10_steps,
                                         hidden_layers=hidden_layers)

    results_path += "tc_3/"
    _write_results_tc_3(ret_val, saved_model_dir, results_path)

    if ret_val == IXO_RET_SUCCESS:
        return saved_model_dir

def _write_results_tc_4(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    result.to_csv(results_path + "tc_results.csv")

    clf_metrics = ClfMetrics()
    cnf_mat, clf_rep = clf_metrics.default_metrics(result['Intent'], result['labels_pred'])
    with open(results_path + "tc_accuracy_report.txt", 'w') as f:
        f.write("confusion_matrix:\n")
        f.write(str(cnf_mat))
        f.write("\n\n")
        f.write("classification_report:\n")
        f.write(str(clf_rep))

    results_incorrect_path = results_path + "result_incorrect.csv"
    result_incorrect = result[result['correct'] == False]
    if result_incorrect.size:
        result_incorrect.to_csv(results_incorrect_path)
    elif os.path.exists(results_incorrect_path):
        os.remove(results_incorrect_path)

def _tc_4(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "intent_test.csv")

    result = tc
    labels_pred = []
    scores_pred = []
    for seq in result['Utterance']:
        ret_val, label_pred, score_pred, _ = model.pred(seq)
        assert(not (isinstance(label_pred, list) or isinstance(score_pred, list)))
        if ret_val == IXO_RET_SUCCESS:
            labels_pred.append(label_pred)
            scores_pred.append(score_pred)
    result['labels_pred'] = labels_pred
    result['scores_pred'] = scores_pred
    result['correct'] = result['Intent'].equals(result['labels_pred'])

    results_path += "tc_4/"
    _write_results_tc_4(result, results_path)

def _write_results_tc_5(ret_val, saved_model_dir, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(results_path + "tc_results.txt", 'w') as f:
        f.write("ret val: ")
        f.write(str(ret_val))
        f.write("\nsaved model dir: ")
        f.write(str(saved_model_dir))

def _tc_5(model, testcases_path, results_path):
    data_path = testcases_path + "imdb_trn"
    data_type = 'txt'
    data_org = 'multi_dir'

    model.config_connectors(data_path=data_path, data_type=data_type, data_org=data_org)

    epochs = 2
    batch_size = 32
    optimizer = 'adam'
    learning_rate = 1e-2
    decay_rate_per_10_steps = 0.0
    hidden_layers = ({'type': 'Dense', 'units': 64, 'activation': 'relu'}, {'type': 'Dropout', 'rate': 0.1})

    ret_val, saved_model_dir = model.trn(epochs=epochs,
                                         batch_size=batch_size,
                                         optimizer=optimizer,
                                         learning_rate=learning_rate,
                                         decay_rate_per_10_steps=decay_rate_per_10_steps,
                                         hidden_layers=hidden_layers)

    results_path += "tc_5/"
    _write_results_tc_5(ret_val, saved_model_dir, results_path)

    if ret_val == IXO_RET_SUCCESS:
        return saved_model_dir

def _write_results_tc_6(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    result.to_csv(results_path + "tc_results.csv")

    clf_metrics = ClfMetrics()
    cnf_mat, clf_rep = clf_metrics.default_metrics(result['Label'], result['labels_pred'])
    with open(results_path + "tc_accuracy_report.txt", 'w') as f:
        f.write("confusion_matrix:\n")
        f.write(str(cnf_mat))
        f.write("\n\n")
        f.write("classification_report:\n")
        f.write(str(clf_rep))

    results_incorrect_path = results_path + "result_incorrect.csv"
    result_incorrect = result[result['correct'] == False]
    if result_incorrect.size:
        result_incorrect.to_csv(results_incorrect_path)
    elif os.path.exists(results_incorrect_path):
        os.remove(results_incorrect_path)

def _tc_6(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "imdb_test.csv")

    result = tc
    labels_pred = []
    scores_pred = []
    for seq in result['Text']:
        ret_val, label_pred, score_pred, _ = model.pred(seq)
        assert(not (isinstance(label_pred, list) or isinstance(score_pred, list)))
        if ret_val == IXO_RET_SUCCESS:
            labels_pred.append(label_pred)
            scores_pred.append(score_pred)
    result['labels_pred'] = labels_pred
    result['scores_pred'] = scores_pred
    result['correct'] = result['Label'].equals(result['labels_pred'])

    results_path += "tc_6/"
    _write_results_tc_6(result, results_path)

def _write_results_tc_7(ret_val, saved_model_dir, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(results_path + "tc_results.txt", 'w') as f:
        f.write("ret val: ")
        f.write(str(ret_val))
        f.write("\nsaved model dir: ")
        f.write(str(saved_model_dir))

def _tc_7(model, testcases_path, results_path):
    data_path = testcases_path + "netflix_trn.csv"
    data_type = 'csv'
    data_org = 'single_file'
    feature_cols = ['description']
    target_cols = ['genres']

    model.config_connectors(data_path=data_path, data_type=data_type, data_org=data_org,
                            feature_cols=feature_cols, target_cols=target_cols)

    epochs = 2

    ret_val, saved_model_dir = model.trn(epochs=epochs)

    results_path += "tc_7/"
    _write_results_tc_7(ret_val, saved_model_dir, results_path)

    if ret_val == IXO_RET_SUCCESS:
        return saved_model_dir

def _write_results_tc_8(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    result.to_csv(results_path + "tc_results.csv")

    clf_metrics = ClfMetrics()
    cnf_mat, clf_rep = clf_metrics.default_metrics(result['genres'], result['labels_pred'], multi_label=True)
    with open(results_path + "tc_accuracy_report.txt", 'w') as f:
        f.write("confusion_matrix:\n")
        f.write(str(cnf_mat))
        f.write("\n\n")
        f.write("classification_report:\n")
        f.write(str(clf_rep))

    results_incorrect_path = results_path + "result_incorrect.csv"
    result_incorrect = result[result['correct'] == False]
    if result_incorrect.size:
        result_incorrect['genres'] = result_incorrect['genres'].astype(str)
        result_incorrect['labels_pred'] = result_incorrect['labels_pred'].astype(str)
        result_incorrect['scores_pred'] = result_incorrect['scores_pred'].astype(str)

        result_incorrect.to_csv(results_incorrect_path)
    elif os.path.exists(results_incorrect_path):
        os.remove(results_incorrect_path)

def _tc_8(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "netflix_test.csv")

    result = tc
    labels_pred = []
    scores_pred = []

    for seq in result['description']:
        ret_val, label_pred, score_pred, pred_dict = model.pred(seq, multi_label=True, proba_thr=0.5, top_n=5)
        assert(isinstance(label_pred, list) and isinstance(score_pred, list))
        if ret_val == IXO_RET_SUCCESS:
            scores_pred.append(score_pred)
            labels_pred.append(label_pred)

    result['labels_pred'] = labels_pred
    result['scores_pred'] = scores_pred

    if isinstance(result['genres'][0], str):
        result['genres'] = result['genres'].apply(lambda x: literal_eval(str(x)))

    result['correct'] = result.apply(lambda row: set(row['genres']) == set(row['labels_pred']), axis=1)

    results_path += "tc_8/"
    _write_results_tc_8(result, results_path)

def _write_results_tc_9(ret_val, saved_model_dir, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(results_path + "tc_results.txt", 'w') as f:
        f.write("ret val: ")
        f.write(str(ret_val))
        f.write("\nsaved model dir: ")
        f.write(str(saved_model_dir))

def _tc_9(model, testcases_path, results_path):
    data_path = testcases_path + "toxic_trn.csv"
    data_type = 'csv'
    data_org = 'single_file'
    feature_cols = ['comment']
    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'normal']

    model.config_connectors(data_path=data_path, data_type=data_type, data_org=data_org,
                            feature_cols=feature_cols, target_cols=target_cols)

    epochs = 2

    ret_val, saved_model_dir = model.trn(epochs=epochs)

    results_path += "tc_9/"
    _write_results_tc_9(ret_val, saved_model_dir, results_path)

    if ret_val == IXO_RET_SUCCESS:
        return saved_model_dir

def _write_results_tc_10(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    result.to_csv(results_path + "tc_results.csv")

    clf_metrics = ClfMetrics()
    cnf_mat, clf_rep = clf_metrics.default_metrics(result['labels'], result['labels_pred'], multi_label=True)
    with open(results_path + "tc_accuracy_report.txt", 'w') as f:
        f.write("confusion_matrix:\n")
        f.write(str(cnf_mat))
        f.write("\n\n")
        f.write("classification_report:\n")
        f.write(str(clf_rep))

    results_incorrect_path = results_path + "result_incorrect.csv"
    result_incorrect = result[result['correct'] == False]
    if result_incorrect.size:
        result_incorrect['labels'] = result_incorrect['labels'].astype(str)
        result_incorrect['labels_pred'] = result_incorrect['labels_pred'].astype(str)
        result_incorrect['scores_pred'] = result_incorrect['scores_pred'].astype(str)

        result_incorrect.to_csv(results_incorrect_path)
    elif os.path.exists(results_incorrect_path):
        os.remove(results_incorrect_path)

def _tc_10(model, testcases_path, results_path):
    tc = pd.read_csv(testcases_path + "toxic_test.csv")

    result = tc
    labels_pred = []
    scores_pred = []

    target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'normal']
    labels = []

    for _, row in result.iterrows():
        seq = row['comment']

        label = []
        for target_col in target_cols:
            if row[target_col]:
                label.append(target_col)
        labels.append(label)

        ret_val, label_pred, score_pred, pred_dict = model.pred(seq, multi_label=True, proba_thr=0.5)
        assert(isinstance(label_pred, list) and isinstance(score_pred, list))
        if ret_val == IXO_RET_SUCCESS:
            scores_pred.append(score_pred)
            labels_pred.append(label_pred)

    result['labels_pred'] = labels_pred
    result['scores_pred'] = scores_pred
    result['labels'] = labels

    result['correct'] = result.apply(lambda row: set(row['labels']) == set(row['labels_pred']), axis=1)

    results_path += "tc_10/"
    _write_results_tc_10(result, results_path)

def _perform_tests():
    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/nlp/text_clf/unit-test/"
    testcases_path = ut_path + "testcases/"
    results_path = ut_path + "results/"

    # Single CSV file
    settings = ['small_data', 'very_light']
    model_trn_single_label = TextClfTrn(settings=settings)
    _, _ = model_trn_single_label.configs.set_gpu(gpu_alloc_in_mb=1024)
    saved_model_dir = _tc_1(model_trn_single_label, testcases_path, results_path)

    model_pred = TextClfPred(saved_model_dir)
    _tc_2(model_pred, testcases_path, results_path)
    
    # Hyper-parameter tuning
    model_trn_hyper_param_tuning = TextClfTrn()
    saved_model_dir = _tc_3(model_trn_hyper_param_tuning, testcases_path, results_path)

    model_pred = TextClfPred(saved_model_dir)
    _tc_4(model_pred, testcases_path, results_path)
    
    # Txt files stored in directory format
    saved_model_dir = _tc_5(model_trn_single_label, testcases_path, results_path)

    model_pred = TextClfPred(saved_model_dir)
    _tc_6(model_pred, testcases_path, results_path)

    # Multi-label (target as a list)
    model_trn_multi_label = TextClfTrn(settings=settings, multi_label=True)
    saved_model_dir = _tc_7(model_trn_multi_label, testcases_path, results_path)

    model_pred = TextClfPred(saved_model_dir)
    _tc_8(model_pred, testcases_path, results_path)

    # Multi-label (target in binary form)
    saved_model_dir = _tc_9(model_trn_multi_label, testcases_path, results_path)

    model_pred = TextClfPred(saved_model_dir)
    _tc_10(model_pred, testcases_path, results_path)

if __name__ == '__main__':
    _perform_tests()

    unittest.main()

    _cleanup()

