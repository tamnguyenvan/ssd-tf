#######################################################################################################################
# Unit-test for Generic Classifier.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kirankumar A M
#######################################################################################################################

from utils.log import Log
from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.metrics.metrics import ClfMetrics
from meghnad.core.std_ml.gen_clf.src.trn import GenericClfTrn
from meghnad.core.std_ml.gen_clf.src.pred import GenericClfPred
from meghnad.core.std_ml.gen_clf.src.pred_prep import GenericClfPredPrep

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import unittest

import os, gc, sys
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)

import warnings
warnings.filterwarnings('ignore')

log = Log()


def _cleanup():
    gc.collect()


def _write_results_tc_1(ret_val, saved_model_dir, results_path, feature_importances):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    with open(results_path + "tc_results.txt", 'w') as f:
        f.write("ret val: ")
        f.write(str(ret_val))
        f.write("\n\nBest performing model on training data: "+os.listdir(saved_model_dir+'/best_model')[0])
        f.write("\n\nBest performing model is saved in dir: ")
        f.write(str(saved_model_dir))
        if feature_importances is not None:
            f.write("\n\nFeature importances of the trained-model: ")
            f.write(str(feature_importances))


def _tc_1(testcases_path, results_path, performance_metric):

    # WA_Fn-UseC_-Telco-Customer-Churn dataset (binary classification)
    data_path = testcases_path + "WA_Fn-UseC_-Telco-Customer-Churn_train.csv"
    data_type = 'csv'
    seperator = ','
    data_org = 'single_dir'
    feature_cols = ['gender', 'tenure', 'MonthlyCharges']
    target_cols = ['Churn']
    multi_targets = False
    multi_labels = False
    cv_folds = 10
    model_mode = 'medium'

    # training data
    clf_train = GenericClfTrn(cv_folds=cv_folds)

    clf_train.config_connectors(data_path=data_path, data_type=data_type, target_cols=target_cols, data_org=data_org,
                                multi_targets=multi_targets, multi_labels=multi_labels, seperator=seperator,
                                feature_cols=feature_cols, model_mode=model_mode)

    ret_val, directory, feature_importances = clf_train.trn(performance_metric)

    results_path += "tc_1/"

    _write_results_tc_1(ret_val, directory, results_path, feature_importances)

    if ret_val == IXO_RET_SUCCESS:
        return directory


def _write_results_tc_2(result, results_path, model_pred, other_pred, metric, actual_value, index):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                "Predicted values of test data (using best performing model on train data) is saved in tc_results.csv")
    result.to_csv(results_path + "tc_results.csv")

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                "Performance of all the models on test data is saved in tc_report.txt")

    performance = ClfMetrics()

    with open(results_path + "tc_report.txt", 'w') as f:
        for mod in model_pred.keys():
            f.write(str(mod))
            f.write("\n\n")
            f.write("confusion_matrix:\n")
            f.write(str(performance.default_metrics(result['Churn'], result['labels_pred'])[0]))
            f.write("\n\n")
            f.write("classification_report:\n")
            f.write(str(performance.default_metrics(result['Churn'], result['labels_pred'])[1])+"\n")
            f.write("===================================================\n")

        for mod in other_pred.keys():
            others_pred = other_pred[mod]['values']['pred']
            result.loc[index, ['others_pred']] = others_pred
            f.write(str(mod))
            f.write("\n\n")
            f.write("confusion_matrix:\n")
            f.write(str(performance.default_metrics(result['Churn'], result['others_pred'])[0]))
            f.write("\n\n")
            f.write("classification_report:\n")
            f.write(str(performance.default_metrics(result['Churn'], result['others_pred'])[1])+"\n")

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                f"{metric} of all the models on test data is saved in model_summary_report.txt")

    with open(results_path + "model_summary_report.txt", 'w') as f:
        for i, mod in enumerate(model_pred.keys()):
            if i == 0:
                f.write("Model --- "+metric+" \n\n")
            f.write(str(mod)+' --- '+str(np.round(performance.user_metrics
                                                  (metric, actual_value,
                                                   model_pred[mod]['values']['model_pred'],
                                                   model_pred[mod]['values']['model_prob'][:, 1])[1], 3))+"\n\n")
        for mod in other_pred.keys():
            f.write(str(mod)+' --- '+str(np.round(performance.user_metrics
                                                  (metric, actual_value,
                                                   other_pred[mod]['values']['model_pred'],
                                                   other_pred[mod]['values']['model_prob'][:, 1])[1], 3))+"\n\n")


def _tc_2(directory, testcases_path, results_path, metric):

    df_data = pd.read_csv(testcases_path + "WA_Fn-UseC_-Telco-Customer-Churn_test.csv")

    # pre-processing tc_data
    df_prep = GenericClfPredPrep(directory, df_data)
    df_, actual_value, index = df_prep.pred_prep()

    # predicting target variable
    clf_pred = GenericClfPred(df_, directory)
    ret_val, model_pred = clf_pred.best_model_prediction()
    other_pred = clf_pred.other_model_prediction()

    if ret_val == IXO_RET_SUCCESS:
        for model in model_pred.keys():
            labels_pred = model_pred[model]['values']['pred']
            df_data.loc[index, ['labels_pred']] = labels_pred

    results_path += "tc_2/"

    _write_results_tc_2(df_data, results_path, model_pred, other_pred, metric, actual_value, index)

# ======================================================================================================================


def _write_results_tc_3(ret_val, saved_model_dir, results_path, feature_importances):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    with open(results_path + "tc_results.txt", 'w') as f:
        f.write("ret val: ")
        f.write(str(ret_val))
        f.write("\n\nBest performing model on training data: "+os.listdir(saved_model_dir+'/best_model')[0])
        f.write("\n\nBest performing model is saved in dir: ")
        f.write(str(saved_model_dir))
        if feature_importances is not None:
            f.write("\n\nFeature importances of the trained-model: ")
            f.write(str(feature_importances))


def _tc_3(testcases_path, results_path, performance_metric):

    # Education dataset (multi-classification)
    data_path = testcases_path + "Edu_train.txt"
    data_type = 'txt'
    seperator = '\t'
    data_org = 'single_dir'
    feature_cols = None
    target_cols = ['Class']
    multi_targets = False
    multi_labels = False
    cv_folds = 10

    # training data
    clf_train = GenericClfTrn(cv_folds=cv_folds)

    clf_train.config_connectors(data_path=data_path, data_type=data_type, target_cols=target_cols, data_org=data_org,
                                multi_targets=multi_targets, multi_labels=multi_labels, seperator=seperator,
                                feature_cols=feature_cols)

    cfg = \
    {
        'models_with_hyper-parameters':
        {
            'LogisticRegression':
            {
                'object': LogisticRegression(),
                'hyper-parameters':
                {
                    'penalty': ['l2'],
                    'fit_intercept': [True]
                }
            },
            'KNearestNeighbour':
            {
                'object': KNeighborsClassifier(),
                'hyper-parameters':
                {
                    'n_neighbors': [5, 50, 100],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan'],
                    'algorithm': ['auto', 'ball_tree']
                }
            },
            'NaiveBayes':
            {
                'object': GaussianNB(),
                'hyper-parameters':
                {
                    'var_smoothing': [2e-9, 5e-8]
                }
            }
        },
        'other_gridsearchcv_params':
        {
            'scoring': 'roc_auc',
            'cv': 2,
            'verbose': 1,
            'n_jobs': None
        }
    }

    ret_val, directory, feature_importances = clf_train.trn(performance_metric, cfg)

    results_path += "tc_3/"

    _write_results_tc_3(ret_val, directory, results_path, feature_importances)

    if ret_val == IXO_RET_SUCCESS:
        return directory


def _write_results_tc_4(result, results_path, model_pred, other_pred, metric, actual_value, index):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                "Predicted values of test data (using best performing model on train data) is saved in tc_results.csv")
    result.to_csv(results_path + "tc_results.csv")

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                "Performance of all the models on test data is saved in tc_report.txt")

    performance = ClfMetrics()

    with open(results_path + "tc_report.txt", 'w') as f:
        for mod in model_pred.keys():
            f.write(str(mod))
            f.write("\n\n")
            f.write("confusion_matrix:\n")
            f.write(str(performance.default_metrics(result['Class'], result['labels_pred'])[0]))
            f.write("\n\n")
            f.write("classification_report:\n")
            f.write(str(performance.default_metrics(result['Class'], result['labels_pred'])[1]) + "\n")
            f.write("===================================================\n")

        for mod in other_pred.keys():
            others_pred = other_pred[mod]['values']['pred']
            result.loc[index, ['others_pred']] = others_pred
            f.write(str(mod))
            f.write("\n\n")
            f.write("confusion_matrix:\n")
            f.write(str(performance.default_metrics(result['Class'], result['others_pred'])[0]))
            f.write("\n\n")
            f.write("classification_report:\n")
            f.write(str(performance.default_metrics(result['Class'], result['others_pred'])[1]) + "\n")

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                f"{metric} of all the models on test data is saved in model_summary_report.txt")

    with open(results_path + "model_summary_report.txt", 'w') as f:
        for i, mod in enumerate(model_pred.keys()):
            if i == 0:
                f.write("Model --- " + metric + " \n\n")
            f.write(str(mod) + ' --- ' + str(np.round(performance.user_metrics
                                                      (metric, actual_value,
                                                       model_pred[mod]['values']['model_pred'],
                                                       model_pred[mod]['values']['model_prob'],
                                                       multi_class='ovr',
                                                       average=None)[1], 3)) + "\n\n")
        for mod in other_pred.keys():
            f.write(str(mod) + ' --- ' + str(np.round(performance.user_metrics
                                                      (metric, actual_value,
                                                       other_pred[mod]['values']['model_pred'],
                                                       other_pred[mod]['values']['model_prob'],
                                                       multi_class='ovr',
                                                       average=None)[1], 3)) + "\n\n")


def _tc_4(directory, testcases_path, results_path, metric):

    df_data = pd.read_csv(testcases_path + "Edu_test.txt", sep='\t')

    # pre-processing tc_data
    df_prep = GenericClfPredPrep(directory, df_data)
    df_, actual_value, index = df_prep.pred_prep()

    # predicting target variable
    clf_pred = GenericClfPred(df_, directory)
    ret_val, model_pred = clf_pred.best_model_prediction()
    other_pred = clf_pred.other_model_prediction()

    if ret_val == IXO_RET_SUCCESS:
        for model in model_pred.keys():
            labels_pred = model_pred[model]['values']['pred']
            df_data.loc[index, ['labels_pred']] = labels_pred

    results_path += "tc_4/"

    _write_results_tc_4(df_data, results_path, model_pred, other_pred, metric, actual_value, index)

# ======================================================================================================================


def _write_results_tc_5(ret_val, saved_model_dir, results_path, feature_importances):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    with open(results_path + "tc_results.txt", 'w') as f:
        f.write("ret val: ")
        f.write(str(ret_val))
        f.write("\n\nBest performing model on training data: "+os.listdir(saved_model_dir+'/best_model')[0])
        f.write("\n\nBest performing model is saved in dir: ")
        f.write(str(saved_model_dir))
        if feature_importances is not None:
            f.write("\n\nFeature importances of the trained-model: ")
            f.write(str(feature_importances))


def _tc_5(testcases_path, results_path, performance_metric):

    # Movie dataset (multi-label dataset)
    data_path = testcases_path + "Movie_train.csv"
    data_type = 'csv'
    seperator = ','
    data_org = 'single_dir'
    feature_cols = None
    target_cols = ['Genre']
    multi_targets = False
    multi_labels = True
    multi_labels_seperator = ','
    cv_folds = 10
    model_mode = 'fast'

    # training data
    clf_train = GenericClfTrn(cv_folds=cv_folds)

    clf_train.config_connectors(data_path=data_path, data_type=data_type, target_cols=target_cols, data_org=data_org,
                                multi_targets=multi_targets, multi_labels=multi_labels, seperator=seperator,
                                feature_cols=feature_cols, multi_labels_seperator=multi_labels_seperator,
                                model_mode=model_mode)

    ret_val, directory, feature_importances = clf_train.trn(performance_metric)

    results_path += "tc_5/"

    _write_results_tc_5(ret_val, directory, results_path, feature_importances)

    if ret_val == IXO_RET_SUCCESS:
        return directory


def _write_results_tc_6(result, results_path, model_pred, other_pred, metric, actual_value, index):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                "Predicted values of test data (using best performing model on train data) is saved in tc_results.csv")
    result.to_csv(results_path + "tc_results.csv")

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                "Performance of all the models on test data is saved in tc_report.txt")

    performance = ClfMetrics()

    with open(results_path + "tc_report.txt", 'w') as f:
        for mod in model_pred.keys():
            f.write(str(mod))
            f.write("\n\n")
            f.write("confusion_matrix:\n")
            f.write(str(performance.default_metrics(result['Genre'], result['labels_pred'], multi_label=True,
                                                    seperator=',')[0]))
            f.write("\n\n")
            f.write("classification_report:\n")
            f.write(str(performance.default_metrics(result['Genre'], result['labels_pred'], multi_label=True,
                                                    seperator=',')[1])+"\n")
            f.write("===================================================\n")

        for mod in other_pred.keys():
            others_pred = other_pred[mod]['values']['pred']
            others_pred = [list(ele) for ele in others_pred]
            result.loc[index, ['others_pred']] = others_pred
            f.write(str(mod))
            f.write("\n\n")
            f.write("confusion_matrix:\n")
            f.write(str(performance.default_metrics(result['Genre'], result['others_pred'], multi_label=True,
                                                    seperator=',')[0]))
            f.write("\n\n")
            f.write("classification_report:\n")
            f.write(str(performance.default_metrics(result['Genre'], result['others_pred'], multi_label=True,
                                                    seperator=',')[1])+"\n")

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                f"{metric} of all the models on test data is saved in model_summary_report.txt")

    with open(results_path + "model_summary_report.txt", 'w') as f:
        for i, mod in enumerate(model_pred.keys()):
            if i == 0:
                f.write("Model --- "+metric+" \n\n")
            f.write(str(mod)+' --- '+str(np.round(performance.user_metrics
                                                  (metric, actual_value,
                                                   model_pred[mod]['values']['model_pred'],
                                                   y_prob=[0.0],
                                                   average='samples')[1], 3))+"\n\n")
        for mod in other_pred.keys():
            f.write(str(mod)+' --- '+str(np.round(performance.user_metrics
                                                  (metric, actual_value,
                                                   other_pred[mod]['values']['model_pred'],
                                                   y_prob=[0.0],
                                                   average='samples')[1], 3))+"\n\n")


def _tc_6(directory, testcases_path, results_path, metric):

    df_data = pd.read_csv(testcases_path + "Movie_test.csv")
    seperator = ','

    # pre-processing tc_data
    df_prep = GenericClfPredPrep(directory, df_data, seperator)
    df_, actual_value, index = df_prep.pred_prep()

    # predicting target variable
    clf_pred = GenericClfPred(df_, directory)
    ret_val, model_pred = clf_pred.best_model_prediction()
    other_pred = clf_pred.other_model_prediction()

    if ret_val == IXO_RET_SUCCESS:
        for model in model_pred.keys():
            labels_pred = model_pred[model]['values']['pred']
            labels_pred = [list(ele) for ele in labels_pred]
            df_data.loc[index, ['labels_pred']] = labels_pred

    results_path += "tc_6/"

    _write_results_tc_6(df_data, results_path, model_pred, other_pred, metric, actual_value, index)

# ======================================================================================================================


def _write_results_tc_7(ret_val, saved_model_dir, results_path, feature_importances):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    with open(results_path + "tc_results.txt", 'w') as f:
        f.write("ret val: ")
        f.write(str(ret_val))
        f.write("\n\nBest performing model on training data: "+os.listdir(saved_model_dir+'/best_model')[0])
        f.write("\n\nBest performing model is saved in dir: ")
        f.write(str(saved_model_dir))
        if feature_importances is not None:
            f.write("\n\nFeature importances of the trained-model: ")
            f.write(str(feature_importances))


def _tc_7(testcases_path, results_path, performance_metric):

    # Education dataset (multi-classification and multi-directory-general)
    data_path = testcases_path + "Edu_train"
    data_type = 'csv'
    seperator = ','
    data_org = 'multi_dir_general'
    feature_cols = ['gender', 'raisedhands', 'StudentAbsenceDays']
    target_cols = ['Class']
    multi_targets = False
    multi_labels = False

    # training data
    clf_train = GenericClfTrn()

    clf_train.config_connectors(data_path=data_path, data_type=data_type, target_cols=target_cols, data_org=data_org,
                                multi_targets=multi_targets, multi_labels=multi_labels, seperator=seperator,
                                feature_cols=feature_cols)

    ret_val, directory, feature_importances = clf_train.trn(performance_metric)

    results_path += "tc_7/"

    _write_results_tc_7(ret_val, directory, results_path, feature_importances)

    if ret_val == IXO_RET_SUCCESS:
        return directory


def _write_results_tc_8(result, results_path, model_pred, other_pred, metric, actual_value, index):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                "Predicted values of test data (using best performing model on train data) is saved in tc_results.csv")
    result.to_csv(results_path + "tc_results.csv")

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                "Performance of all the models on test data is saved in tc_report.txt")

    performance = ClfMetrics()

    with open(results_path + "tc_report.txt", 'w') as f:
        for mod in model_pred.keys():
            f.write(str(mod))
            f.write("\n\n")
            f.write("confusion_matrix:\n")
            f.write(str(performance.default_metrics(result['Class'], result['labels_pred'])[0]))
            f.write("\n\n")
            f.write("classification_report:\n")
            f.write(str(performance.default_metrics(result['Class'], result['labels_pred'])[1])+"\n")
            f.write("===================================================\n")

        for mod in other_pred.keys():
            others_pred = other_pred[mod]['values']['pred']
            result.loc[index, ['others_pred']] = others_pred
            f.write(str(mod))
            f.write("\n\n")
            f.write("confusion_matrix:\n")
            f.write(str(performance.default_metrics(result['Class'], result['others_pred'])[0]))
            f.write("\n\n")
            f.write("classification_report:\n")
            f.write(str(performance.default_metrics(result['Class'], result['others_pred'])[1])+"\n")

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                f"{metric} of all the models on test data is saved in model_summary_report.txt")

    with open(results_path + "model_summary_report.txt", 'w') as f:
        for i, mod in enumerate(model_pred.keys()):
            if i == 0:
                f.write("Model --- " + metric + " \n\n")
            f.write(str(mod) + ' --- ' + str(np.round(performance.user_metrics
                                                      (metric, actual_value,
                                                       model_pred[mod]['values']['model_pred'],
                                                       model_pred[mod]['values']['model_prob'],
                                                       multi_class='ovr',
                                                       average=None)[1], 3)) + "\n\n")
        for mod in other_pred.keys():
            f.write(str(mod) + ' --- ' + str(np.round(performance.user_metrics
                                                      (metric, actual_value,
                                                       other_pred[mod]['values']['model_pred'],
                                                       other_pred[mod]['values']['model_prob'],
                                                       multi_class='ovr',
                                                       average=None)[1], 3)) + "\n\n")


def _tc_8(directory, testcases_path, results_path, metric):

    df_data = pd.read_csv(testcases_path + "Edu_test.txt", sep='\t')

    # pre-processing tc_data
    df_prep = GenericClfPredPrep(directory, df_data)
    df_, actual_value, index = df_prep.pred_prep()

    # predicting target variable
    clf_pred = GenericClfPred(df_, directory)
    ret_val, model_pred = clf_pred.best_model_prediction()
    other_pred = clf_pred.other_model_prediction()

    if ret_val == IXO_RET_SUCCESS:
        for model in model_pred.keys():
            labels_pred = model_pred[model]['values']['pred']
            df_data.loc[index, ['labels_pred']] = labels_pred

    results_path += "tc_8/"

    _write_results_tc_8(df_data, results_path, model_pred, other_pred, metric, actual_value, index)

# ======================================================================================================================


def _write_results_tc_9(ret_val, saved_model_dir, results_path, feature_importances):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    with open(results_path + "tc_results.txt", 'w') as f:
        f.write("ret val: ")
        f.write(str(ret_val))
        f.write("\n\nBest performing model on training data: "+os.listdir(saved_model_dir+'/best_model')[0])
        f.write("\n\nBest performing model is saved in dir: ")
        f.write(str(saved_model_dir))
        if feature_importances is not None:
            f.write("\n\nFeature importances of the trained-model: ")
            f.write(str(feature_importances))


def _tc_9(testcases_path, results_path, performance_metric):

    # WA_Fn-UseC_-Telco-Customer-Churn dataset (binary classification and multi-directory-periodic)
    data_path = testcases_path + "Telco_train_periodic"
    data_type = 'csv'
    seperator = ','
    data_org = 'multi_dir_periodic'
    feature_cols = ['gender', 'tenure', 'MonthlyCharges']
    target_cols = ['Churn']
    multi_targets = False
    multi_labels = False
    cv_folds = 10
    model_mode = 'fast'

    # training data
    clf_train = GenericClfTrn(cv_folds=cv_folds)

    clf_train.config_connectors(data_path=data_path, data_type=data_type, target_cols=target_cols, data_org=data_org,
                                multi_targets=multi_targets, multi_labels=multi_labels, seperator=seperator,
                                feature_cols=feature_cols, model_mode=model_mode)

    ret_val, directory, feature_importances = clf_train.trn(performance_metric)

    results_path += "tc_9/"

    _write_results_tc_9(ret_val, directory, results_path, feature_importances)

    if ret_val == IXO_RET_SUCCESS:
        return directory


def _write_results_tc_10(result, results_path, model_pred, other_pred, metric, actual_value, index):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                "Predicted values of test data (using best performing model on train data) is saved in tc_results.csv")
    result.to_csv(results_path + "tc_results.csv")

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                "Performance of all the models on test data is saved in tc_report.txt")

    performance = ClfMetrics()

    with open(results_path + "tc_report.txt", 'w') as f:
        for mod in model_pred.keys():
            f.write(str(mod))
            f.write("\n\n")
            f.write("confusion_matrix:\n")
            f.write(str(performance.default_metrics(result['Churn'], result['labels_pred'])[0]))
            f.write("\n\n")
            f.write("classification_report:\n")
            f.write(str(performance.default_metrics(result['Churn'], result['labels_pred'])[1])+"\n")
            f.write("===================================================\n")

        for mod in other_pred.keys():
            others_pred = other_pred[mod]['values']['pred']
            result.loc[index, ['others_pred']] = others_pred
            f.write(str(mod))
            f.write("\n\n")
            f.write("confusion_matrix:\n")
            f.write(str(performance.default_metrics(result['Churn'], result['others_pred'])[0]))
            f.write("\n\n")
            f.write("classification_report:\n")
            f.write(str(performance.default_metrics(result['Churn'], result['others_pred'])[1])+"\n")

    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                f"{metric} of all the models on test data is saved in model_summary_report.txt")

    with open(results_path + "model_summary_report.txt", 'w') as f:
        for i, mod in enumerate(model_pred.keys()):
            if i == 0:
                f.write("Model --- " + metric + " \n\n")
            f.write(str(mod) + ' --- ' + str(np.round(performance.user_metrics
                                                      (metric, actual_value,
                                                       model_pred[mod]['values']['model_pred'],
                                                       model_pred[mod]['values']['model_prob'][:, 1])[1], 3)) + "\n\n")
        for mod in other_pred.keys():
            f.write(str(mod) + ' --- ' + str(np.round(performance.user_metrics
                                                      (metric, actual_value,
                                                       other_pred[mod]['values']['model_pred'],
                                                       other_pred[mod]['values']['model_prob'][:, 1])[1], 3)) + "\n\n")


def _tc_10(directory, testcases_path, results_path, metric):

    df_data = pd.read_csv(testcases_path + "WA_Fn-UseC_-Telco-Customer-Churn_test.csv")

    # pre-processing tc_data
    df_prep = GenericClfPredPrep(directory, df_data)
    df_, actual_value, index = df_prep.pred_prep()

    # predicting target variable
    clf_pred = GenericClfPred(df_, directory)
    ret_val, model_pred = clf_pred.best_model_prediction()
    other_pred = clf_pred.other_model_prediction()

    if ret_val == IXO_RET_SUCCESS:
        for model in model_pred.keys():
            labels_pred = model_pred[model]['values']['pred']
            df_data.loc[index, ['labels_pred']] = labels_pred

    results_path += "tc_10/"

    _write_results_tc_10(df_data, results_path, model_pred, other_pred, metric, actual_value, index)

# ======================================================================================================================


def _perform_tests():

    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/std_ml/gen_clf/unit-test/"
    # ut_path = str(Path(os.path.dirname('trn-ut.py')).parent.absolute())[:-3]

    results_path = ut_path + "results/"
    testcases_path = ut_path + "testcases/"

    performance_metric = 'roc_auc_score'

    saved_model_dir = _tc_1(testcases_path, results_path, performance_metric)

    _tc_2(saved_model_dir, testcases_path, results_path, performance_metric)

#
# # ====================================================================================================================
#

    performance_metric = 'balanced_accuracy'

    saved_model_dir = _tc_3(testcases_path, results_path, performance_metric)

    _tc_4(saved_model_dir, testcases_path, results_path, performance_metric)
#
# # ====================================================================================================================
#

    performance_metric = 'precision'

    saved_model_dir = _tc_5(testcases_path, results_path, performance_metric)

    _tc_6(saved_model_dir, testcases_path, results_path, performance_metric)

#
# # ====================================================================================================================
#

    performance_metric = 'recall'

    saved_model_dir = _tc_7(testcases_path, results_path, performance_metric)

    _tc_8(saved_model_dir, testcases_path, results_path, performance_metric)
#
# # ====================================================================================================================
#

    performance_metric = 'f1'

    saved_model_dir = _tc_9(testcases_path, results_path, performance_metric)

    _tc_10(saved_model_dir, testcases_path, results_path, performance_metric)

# # ====================================================================================================================


if __name__ == '__main__':

    _perform_tests()

    unittest.main()

    _cleanup()
