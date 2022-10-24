#######################################################################################################################
# Prediction for Generic classifier.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kirankumar A M
#######################################################################################################################

# Import Libraries
import os, sys
from utils.log import Log
from utils.common_defs import *
from utils.ret_values import *
from joblib import load

log = Log()


@class_header(
    description='''
        Generic classifier prediction pipeline.''')
class GenericClfPred():
    def __init__(self, pred_df: object, directory: str):
        self.pred_df = pred_df
        self.directory = directory
        self.encoder = load(self.directory+'pred_arguments/encoder.joblib')
        self.mlb = load(self.directory+'pred_arguments/mlb.joblib')
        for file in os.listdir(self.directory + 'best_model/'):
            self.best_model = load(self.directory + 'best_model/' + file)
            self.best_model_name = file.split('.')[0]
        self.other_models = {}
        for file in os.listdir(self.directory):
            if file.endswith(".joblib"):
                self.other_model = load(self.directory+file)
                self.other_model_name = file.split('.')[0]
                self.other_models[self.other_model_name] = self.other_model

    @method_header(
        description='''
            Prediction of pred_df data using best model.''',
        returns='''
            a 2 member tuple having the IXO return value, and the best model's predicted values.''')
    def best_model_prediction(self) -> (int, dict):
        res_best = {}
        model_pred = self.best_model.predict(self.pred_df)
        model_prob = self.best_model.predict_proba(self.pred_df)
        log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                    f"{self.best_model_name} is used to predict values of validation data")
        try:
            val_pred = self.encoder.inverse_transform(model_pred)
        except:
            val_pred = self.mlb.inverse_transform(model_pred)

        # dictionary to store the actual and predicted value of all models
        res_best[self.best_model_name] = {}
        res_best[self.best_model_name]['values'] = \
            {
                'pred': val_pred,
                'model_prob': model_prob,
                'model_pred': model_pred
            }
        res_best[self.best_model_name]['object'] = self.best_model

        return IXO_RET_SUCCESS, res_best

    @method_header(
        description='''
            Prediction of pred_df data using other models.''',
        returns='''
            other model's predicted values in a dictionary.''')
    def other_model_prediction(self) -> dict:
        res_other = {}
        for model in self.other_models:
            model_pred = self.other_models[model].predict(self.pred_df)
            model_prob = self.other_models[model].predict_proba(self.pred_df)
            log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                        f"{model} is used to predict values of validation data")
            try:
                val_pred = self.encoder.inverse_transform(model_pred)
            except:
                val_pred = self.mlb.inverse_transform(model_pred)

            # dictionary to store the actual and predicted value of all models
            res_other[model] = {}
            res_other[model]['values'] = \
                {
                    'pred': val_pred,
                    'model_prob': model_prob,
                    'model_pred': model_pred
                }
            res_other[model]['object'] = self.other_models[model]

        return res_other


if __name__ == '__main__':
    pass
