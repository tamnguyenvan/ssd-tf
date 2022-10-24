#######################################################################################################################
# Training for Generic classifier.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kirankumar A M
#######################################################################################################################

# Import Libraries

from utils.log import Log
from utils.common_defs import *
from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.std_ml.gen_clf.src.trn_prep import GenericClfTrnPrep
from meghnad.core.std_ml.gen_clf.cfg.config import GenericClfConfig
from meghnad.metrics.metrics import ClfMetrics
import numpy as np
from joblib import dump
import os, shutil, sys

# model selection
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from skmultilearn.model_selection import iterative_train_test_split

log = Log()


@class_header(
    description='''
    Generic classifier training pipeline.''')
class GenericClfTrn():
    def __init__(self, *args, **kwargs):
        self.configs = GenericClfConfig(MeghnadConfig())
        if 'cv_folds' in kwargs:
            self.cv_folds = kwargs['cv_folds']
        else:
            self.cv_folds = self.configs.get_other_gridsearchcv_params()['cv']

    @method_header(
        description='''
            Helper for configuring data connectors.''',
        arguments='''
            data_path: location of the training data (point to the file),
            data_type: type of the training data (csv/excel/txt),
            target_cols: attribute names in the data to be used as targets during training,
            feature_cols: None (if all columns are to be considered else list the required features),
            dat_org: organization of the data (single directory / multi-directory-general/ multi-directory-periodic
                     training data),
            multi_targets: If target columns are more than 1 in un-processed data then True else False
                           (currently not supported),
            multi_labels: If the target variable has multi-labels then True else False,
            seperator: Seperator used in the training & validation file ('\t, , ,'),
            multi_labels_seperator: If multi-labels, seperator of labels present in data,
            dir_to_save_model: Directory where the model needs to be saved.
            model_mode: Training mode ('slow', 'medium', 'fast'), 
                        Decides the number of models to be selected while training.''')
    def config_connectors(self, data_path: str, data_type: str, target_cols: [str], feature_cols: [str],
                          data_org: str,
                          multi_targets: bool,
                          multi_labels: bool,
                          seperator: str,
                          multi_labels_seperator: str = None,
                          dir_to_save_model: str = None,
                          model_mode: str = 'slow'):
        self.connector_trn = {}
        if data_path:
            self.connector_trn['data_path'] = data_path
        else:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      "Connector not configured with any data")
            return IXO_RET_INCORRECT_CONFIG

        if data_type:
            self.connector_trn['data_type'] = data_type
        else:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      "Connector not configured with a data type")
            return IXO_RET_INCORRECT_CONFIG

        if self.connector_trn['data_type'] not in ['csv', 'excel', 'txt']:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__, "Not Implemented")
            return IXO_RET_NOT_IMPLEMENTED

        if target_cols:
            self.connector_trn['target_cols'] = target_cols
        else:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      "Connector not configured with target_cols")
            return IXO_RET_INCORRECT_CONFIG
        if not isinstance(self.connector_trn['target_cols'], list):
            log.ERROR(sys._getframe().f_lineno, __file__, __name__, "Specify the target cols in a list")
            return IXO_RET_INCORRECT_CONFIG

        if feature_cols is None or isinstance(feature_cols, list):
            self.connector_trn['feature_cols'] = feature_cols
        else:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      "Connector not configured with feature_cols")
            return IXO_RET_INCORRECT_CONFIG
        if self.connector_trn['feature_cols'] is None:
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__, "All columns are considered as features")
        elif not isinstance(self.connector_trn['feature_cols'], list):
            log.ERROR(sys._getframe().f_lineno, __file__, __name__, "Specify the feature cols in a list")
            return IXO_RET_INCORRECT_CONFIG

        if data_org:
            self.connector_trn['data_org'] = data_org
        else:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      "Connector not configured with data_org")
            return IXO_RET_INCORRECT_CONFIG
        if self.connector_trn['data_org'] not in ['single_dir', 'multi_dir_general', 'multi_dir_periodic']:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__, "Not Implemented")
            return IXO_RET_NOT_IMPLEMENTED

        self.connector_trn['multi_targets'] = multi_targets
        if self.connector_trn['multi_targets']:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__, "Not Implemented")
            return IXO_RET_NOT_IMPLEMENTED

        if not multi_labels:
            self.connector_trn['multi_labels'] = multi_labels
        else:
            self.connector_trn['multi_labels'] = multi_labels
            self.connector_trn['multi_labels_seperator'] = multi_labels_seperator
            if self.connector_trn['multi_labels_seperator'] is None:
                log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                          "Connector not configured with multi_labels_seperator")
                return IXO_RET_INCORRECT_CONFIG
        if not isinstance(self.connector_trn['multi_labels'], bool):
            log.ERROR(sys._getframe().f_lineno, __file__, __name__, "Connector not configured with multi_labels")
            return IXO_RET_INCORRECT_CONFIG

        if seperator:
            self.connector_trn['seperator'] = seperator
        else:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      "Connector not configured with seperator")
            return IXO_RET_INCORRECT_CONFIG
        if self.connector_trn['seperator'] not in [',', '\t']:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__, "Not Implemented")
            return IXO_RET_NOT_IMPLEMENTED

        if dir_to_save_model:
            self.connector_trn['dir_to_save_model'] = dir_to_save_model
        else:
            dir_to_save_model = MeghnadConfig().get_meghnad_configs('INT_PATH') + 'gen_clf/'
            self.connector_trn['dir_to_save_model'] = dir_to_save_model

        if not os.path.exists(self.connector_trn['dir_to_save_model']):
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__, "Creating directories to save model")
            os.makedirs(self.connector_trn['dir_to_save_model'])
        else:
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                        "Removing existing directory and creating a new one to save model")
            shutil.rmtree(self.connector_trn['dir_to_save_model'])
            os.makedirs(self.connector_trn['dir_to_save_model'])

        self.connector_trn['model_mode'] = model_mode
        log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                    f"{self.connector_trn['model_mode']} training mode is considered.")
        if self.connector_trn['model_mode'] not in ['slow', 'medium', 'fast']:
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      "Not Implemented. Choose any one of ['medium', 'fast']")
            return IXO_RET_NOT_IMPLEMENTED

        if self.connector_trn['target_cols']:
            if self.connector_trn['multi_targets']:
                self.connector_trn['target_cols'] = self.connector_trn['target_cols']
            else:
                self.connector_trn['target_cols'] = self.connector_trn['target_cols'][0]

        if not os.path.exists(self.connector_trn['dir_to_save_model']+'pred_arguments'):
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                        "Creating directory to save arguments to be used for prediction")
            os.mkdir(self.connector_trn['dir_to_save_model']+'pred_arguments')

        if not os.path.exists(self.connector_trn['dir_to_save_model']+'best_model'):
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                        "Creating directory to save the best model to be used for prediction")
            os.mkdir(self.connector_trn['dir_to_save_model']+'best_model')

        # instantiating the training data pre-process class
        clf_prep = GenericClfTrnPrep(self.connector_trn)

        self.train_df, self.target_list, self.drop, self.features_list, self.categorical_variables, \
            self.target, self.mlb, self.encoder = clf_prep.prep()

        self.predictors = [x for x in self.train_df.columns if x not in self.target_list]

        # dumping the pre-processed train data features
        dump(self.features_list, self.connector_trn['dir_to_save_model']+'pred_arguments/initial_features.joblib')
        dump(self.predictors, self.connector_trn['dir_to_save_model']+'pred_arguments/final_features.joblib')
        dump(self.drop, self.connector_trn['dir_to_save_model'] + 'pred_arguments/drop.joblib')
        dump(self.categorical_variables, self.connector_trn['dir_to_save_model'] +
             'pred_arguments/categorical_variables.joblib')
        dump(self.target, self.connector_trn['dir_to_save_model'] + 'pred_arguments/target.joblib')
        dump(self.mlb, self.connector_trn['dir_to_save_model'] + 'pred_arguments/mlb.joblib')
        dump(self.encoder, self.connector_trn['dir_to_save_model'] + 'pred_arguments/encoder.joblib')

    @method_header(
        description='''
            training of models.''',
        arguments='''
            alg: set of models to be used for training
            perf_metric: performance metric on which the model needs to be evaluated.''',
        returns='''
            a 3 member tuple containing the IXO return value, the directory to save model and the 
            feature importances dictionary of the best model.''')
    def trn(self, perf_metric: str, alg: dict = None) -> (int, str, dict):

        performance = ClfMetrics()

        alg = _tune_hyper_params(alg, self.configs, self.train_df, self.connector_trn['model_mode'])

        ret, perf = _model_training(alg, self.connector_trn, self.train_df, self.configs, self.predictors,
                                    self.target_list, performance, self.cv_folds, perf_metric)

        ret, best_model, feature_importances = _model_assessment(perf, self.connector_trn, self.predictors)

        return ret, self.connector_trn['dir_to_save_model'], feature_importances


def _tune_hyper_params(alg: dict, configs: object, train: object, mode: str) -> dict:
    if alg:
        log.VERBOSE(sys._getframe().f_lineno, __file__, __name__, "Training data with user specified models")
        for params in alg['other_gridsearchcv_params'].keys():
            if params not in list(configs.get_other_gridsearchcv_params().keys()):
                log.ERROR(sys._getframe().f_lineno, __file__, __name__, f"{params} cannot be user specified")
                return IXO_RET_NOT_SUPPORTED

        for params in configs.get_other_gridsearchcv_params().keys():
            if params not in list(alg['other_gridsearchcv_params'].keys()):
                log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                          f"please specify {params} in other_gridsearchcv_params")
                return IXO_RET_NOT_SUPPORTED
    else:
        log.VERBOSE(sys._getframe().f_lineno, __file__, __name__, "Training data with default models")
        alg = {'models_with_hyper-parameters': configs.get_models_by_modes(mode),
               'other_gridsearchcv_params': configs.get_other_gridsearchcv_params(),
               'other_general_params': configs.get_other_general_params()}
        # tuning the default hyperparameter based on the size of train data
        if train.shape[0] > alg['other_general_params']['std_data_size']:
            factor = int(round(train.shape[0] / alg['other_general_params']['std_data_size'], 0))
            if factor > configs.get_other_general_params()['max_factor']:
                factor = configs.get_other_general_params()['max_factor']
            for model in alg['models_with_hyper-parameters']:
                for key in alg['models_with_hyper-parameters'][model]['hyper-parameters']:
                    param = alg['models_with_hyper-parameters'][model]['hyper-parameters'][key]
                    if type(param[0]) != bool and type(param[0]) != str:
                        alg['models_with_hyper-parameters'][model]['hyper-parameters'][key] = \
                            [x * factor for x in
                             alg['models_with_hyper-parameters'][model]['hyper-parameters'][key]]
    return alg


def _model_training(alg: dict, connector_trn: dict, train_df: object, configs: object, predictors: list,
                    target_list: [str], performance: object, cv_folds: int, perf_metric: str) -> (int, dict):
    perf = {}

    if train_df.shape[0] <= configs.get_other_general_params()['min_data_size']:
        val_size = configs.get_other_general_params()['min_val_split']
    else:
        val_size = configs.get_other_general_params()['min_val_split'] * \
                   (train_df.shape[0] / configs.get_other_general_params()['min_data_size'])

    if connector_trn['multi_labels']:
        train_X, train_y, val_X, val_y = iterative_train_test_split(train_df[predictors].values,
                                                                    train_df[target_list].values,
                                                                    test_size=max(val_size, configs.
                                                                    get_other_general_params()['max_val_split']))
    else:
        train_X, val_X, train_y, val_y = train_test_split(train_df[predictors], train_df[target_list],
                                                          test_size=max(val_size, configs.
                                                          get_other_general_params()['max_val_split']),
                                                          random_state=IXO_SEED)

    for i in alg['models_with_hyper-parameters'].keys():
        log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                    f"Model {i}")
        # Using GridSearchCV to get the best parameters
        model = alg['models_with_hyper-parameters'][i]['object']
        if connector_trn['multi_labels']:
            model = OneVsRestClassifier(model)
        random_grid = alg['models_with_hyper-parameters'][i]['hyper-parameters']
        if connector_trn['multi_labels']:
            random_grid = {f"estimator__{key}": value for key, value in random_grid.items()}
        model = GridSearchCV(estimator=model, param_grid=random_grid,
                             scoring=alg['other_gridsearchcv_params']['scoring'],
                             cv=alg['other_gridsearchcv_params']['cv'],
                             verbose=alg['other_gridsearchcv_params']['verbose'],
                             n_jobs=alg['other_gridsearchcv_params']['n_jobs'])
        model.fit(train_X, train_y)
        model = model.best_estimator_
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   f"Best hyper-parameter for {i} is: {model}")
        if not connector_trn['multi_labels']:
            # perform stratified cross validation
            skf = StratifiedKFold(n_splits=cv_folds)               # specify number of stratified folds
            y = train_df[target_list].to_numpy()                   # convert target variable to numpy array
            y_pred = y.copy()
            X = train_df[predictors].to_numpy().astype(np.float)   # convert features to numpy array
            cv_score = []
            for ii, jj in skf.split(X, y):                         # perform the stratified split and get indices
                X_train, X_val = X[ii], X[jj]
                y_train = y[ii]
                model.fit(X_train, y_train)
                y_pred[jj] = model.predict(X_val).flatten()
                if train_y.nunique() <= 2:
                    metric_calc = performance.user_metrics(perf_metric, y[jj], y_pred[jj],
                                                           model.predict_proba(X_val)[:, 1])
                    if metric_calc[0] != IXO_RET_SUCCESS:
                        return metric_calc[0], None
                    cv_score.append(metric_calc[1])
                else:
                    metric_calc = performance.user_metrics(perf_metric, y[jj], y_pred[jj], model.predict_proba(X_val),
                                                           multi_class='ovr', average=None)
                    if metric_calc[0] != IXO_RET_SUCCESS:
                        return metric_calc[0], None
                    cv_score.append(metric_calc[1])
            # Stratified cross validation report:
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                        "**Stratified cross validation report**")
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                        "CV_Score: Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g"
                        % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
            # Classification Report
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                        f"Classification Report :\n {performance.default_metrics(y, y_pred)[1]}")
            grad_ens_conf_matrix = performance.default_metrics(y, y_pred)[0]
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                        f"Confusion Matrix : \n {grad_ens_conf_matrix}")

        else:
            y_pred = model.predict(val_X)
            metric_calc = performance.user_metrics(perf_metric, val_y, y_pred, y_prob=[0.0], average='samples')
            if metric_calc[0] != IXO_RET_SUCCESS:
                return metric_calc[0], None
            cv_score = metric_calc[1]
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                        f"Performance of the model on validation data: {cv_score}")

        perf[i] = {}
        perf[i]['cv_score'] = np.mean(cv_score)
        perf[i]['object'] = model

    return IXO_RET_SUCCESS, perf


def _model_assessment(perf: dict, connector_trn: dict, predictors: [str]) -> (int, str, dict):
    max_cv_model = max(perf, key=lambda x: perf[x]['cv_score'])
    log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                f"Best performing model: {max_cv_model}")
    for mod in perf.keys():
        file_name = mod + '.joblib'
        if mod == max_cv_model:
            dump(perf[mod]['object'], connector_trn['dir_to_save_model']+'best_model/'+file_name)
            try:
                feature_importances = perf[mod]['object'].feature_importances_
                feature_importance = dict(zip(predictors, feature_importances))
                sorted_feature_importances = dict(sorted(feature_importance.items(),
                                                         key=lambda x: x[1], reverse=True))
            except:
                sorted_feature_importances = None
        else:
            dump(perf[mod]['object'], connector_trn['dir_to_save_model']+file_name)

    log.STATUS(sys._getframe().f_lineno, __file__, __name__,
               f"Best performing model was saved in the directory: {connector_trn['dir_to_save_model']}best_model")
    log.STATUS(sys._getframe().f_lineno, __file__, __name__,
               f"Other models are saved in the directory: {connector_trn['dir_to_save_model']}")

    return IXO_RET_SUCCESS, max_cv_model, sorted_feature_importances


if __name__ == '__main__':
    pass
