#######################################################################################################################
# Config for Generic classifier.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kirankumar A M
#######################################################################################################################

# Import Libraries

from utils.log import Log
from utils.common_defs import *
from meghnad.cfg.config import MeghnadConfig
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn import svm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

log = Log()

# Default Models, hyper-parameters, other parameters to be used for Grid Search and other general parameters of data
_default_clf_cfg = \
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
                'metric': ['minkowski', 'euclidean', 'manhattan']
            }
        },
        'RandomForest':
        {
            'object': ensemble.RandomForestClassifier(),
            'hyper-parameters':
            {
                'n_estimators': [50, 100, 200],
                'max_features': ['auto', 'sqrt'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 4, 10]
            }
        },
        'AdaboostClassifier':
        {
            'object': ensemble.AdaBoostClassifier(),
            'hyper-parameters':
            {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1]
            }
        },
        'SupportVectorMachines':
        {
            'object': svm.SVC(),
            'hyper-parameters':
            {
                'C': [0.01, 0.05, 0.1, 5],
                'kernel': ['linear', 'poly', 'rbf'],
                'probability': [True]
            }
        },
        'GBM':
        {
            'object': ensemble.GradientBoostingClassifier(),
            'hyper-parameters':
            {
                'loss': ['log_loss', 'deviance', 'exponential'],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200],
                'max_features': ['auto', 'sqrt'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 4, 10]
            }
        },
        'Xgboost':
        {
            'object': XGBClassifier(),
            'hyper-parameters':
            {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'num_parallel_tree': [3, 6, 9],
                'min_child_weight': [1, 3, 5]
            }
        },
        'LGBM':
        {
            'object': LGBMClassifier(),
            'hyper-parameters':
            {
                'boosting_type': ['gbdt', 'dart', 'goss'],
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5]
            }
        },
        'Catboost':
        {
            'object': CatBoostClassifier(),
            'hyper-parameters':
                {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7]
                }
        },
        'MLP':
        {
            'object': MLPClassifier(),
            'hyper-parameters':
                {
                    'hidden_layer_sizes': [(25,), (50,), (50, 25)],
                    'activation': ['logistic', 'tanh', 'relu'],
                    'learning_rate_init': [0.001, 0.01, 0.05],
                    'solver': ['lbfgs', 'sgd', 'adam'],
                    'batch_size': ['auto', 8, 16],
                    'alpha': [0.0001, 0.0005, 0.001],
                    'max_iter': [10, 15]
                }
        }
    },
    'other_gridsearchcv_params':
    {
        'scoring': 'roc_auc',
        'cv': 2,
        'verbose': IXO_LOG_VERBOSE,
        'n_jobs': None
    },
    'other_general_params':
    {
        'min_val_split': 0.10,
        'max_val_split': 0.45,
        'min_data_size': 10000,
        'std_data_size': 25000,
        'max_factor': 5
    }
}


class GenericClfConfig(MeghnadConfig):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_models_by_modes(self, mode: str) -> dict:
        models = {}
        if mode == 'slow':
            for name in _default_clf_cfg['models_with_hyper-parameters']:
                models[name] = _default_clf_cfg['models_with_hyper-parameters'][name]
        elif mode == 'medium':
            for name in _default_clf_cfg['models_with_hyper-parameters']:
                if name not in ['Xgboost', 'MLP', 'GBM', 'Catboost']:
                    models[name] = _default_clf_cfg['models_with_hyper-parameters'][name]
        else:
            for name in _default_clf_cfg['models_with_hyper-parameters']:
                if name in ['LogisticRegression', 'KNearestNeighbour', 'SupportVectorMachines']:
                    models[name] = _default_clf_cfg['models_with_hyper-parameters'][name]
        return models

    def get_other_gridsearchcv_params(self) -> dict:
        params = {}
        for param in _default_clf_cfg['other_gridsearchcv_params']:
            params[param] = _default_clf_cfg['other_gridsearchcv_params'][param]
        return params

    def get_other_general_params(self) -> dict:
        params = {}
        for param in _default_clf_cfg['other_general_params']:
            params[param] = _default_clf_cfg['other_general_params'][param]
        return params


if __name__ == '__main__':
    pass
