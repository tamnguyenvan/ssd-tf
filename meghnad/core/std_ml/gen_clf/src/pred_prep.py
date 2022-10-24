#######################################################################################################################
# Preprocessing of prediction data for Generic Classifier.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kirankumar A M
#######################################################################################################################

# Import Libraries
from utils.log import Log
from utils.common_defs import *
import pandas as pd
import numpy as np
import sys
from joblib import load
from ast import literal_eval

# pre-processing modules
from sklearn import preprocessing


log = Log()


@class_header(
    description='''
    Generic classifier pred_df data preprocessing pipeline.''')
class GenericClfPredPrep():
    def __init__(self, directory: str, data: object, seperator: str = None):

        self.directory = directory
        self.drop = load(directory + 'pred_arguments/drop.joblib')
        self.final_features = load(directory + 'pred_arguments/final_features.joblib')
        self.initial_features = load(directory + 'pred_arguments/initial_features.joblib')
        self.target = load(directory + 'pred_arguments/target.joblib')
        self.categorical_variables = load(directory + 'pred_arguments/categorical_variables.joblib')
        self.mlb = load(directory + 'pred_arguments/mlb.joblib')
        self.target_encoder = load(directory + 'pred_arguments/encoder.joblib')
        self.pred_df = pd.concat([data[self.initial_features], data[self.target]], axis=1)
        self.seperator = seperator

    @method_header(
        description='''
            Pre-processing pred_df data.''',
        returns='''
            a 3 member tuple having the pre-processed pred_df data, actual value of the target variable before 
            pre-processing, and the index value of rows with no missing values in pred_df data.
            ''')
    def pred_prep(self) -> (object, [int], [int]):

        self.get_data()
        self.one_hot_encoding()
        self.standardize()

        return self.pred_df, self.actual_value, self.index

    @method_header(
        description='''
            initial pre-processing of pred_df data.''')
    def get_data(self):

        self.pred_df.drop(self.drop, axis=1, inplace=True)
        # index values
        initial_index = self.pred_df.index.values

        # replacing empty strings and blank spaces in data to Nan
        self.pred_df.replace('', np.nan, inplace=True)
        self.pred_df.replace(' ', np.nan, inplace=True)
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "Replacing empty strings and blank spaces in data to Nan")

        # convert those empty string replaced columns from object dtype to float dtype based on other values
        for i in self.pred_df.columns:
            if (self.pred_df[i].dtype == 'O') and (self.pred_df[i].isna().sum() > 0):
                try:
                    self.pred_df[i] = self.pred_df[i].astype(float)
                    log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                               "Empty string replaced to Nan columns are changed from object to float data type")
                except ValueError:
                    continue

        # getting the index of rows with missing values
        final_index = self.pred_df[self.pred_df.isnull().any(axis=1)].index.values
        # only those indices with rows not having any missing values
        self.index = np.delete(initial_index, final_index)
        # dropping rows with missing values
        self.pred_df.dropna(axis=0, inplace=True)
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "Dropping rows with missing values in pred_df data")

    @method_header(
        description='''
            one hot encoding the categorical variables in pred_df data (including target).''')
    def one_hot_encoding(self):

        if self.mlb is not None:
            try:
                self.pred_df[self.target] = self.pred_df[self.target].apply(lambda x: literal_eval(str(x)))
            except:
                self.pred_df[self.target] = self.pred_df[self.target].apply(lambda x: x.split(self.seperator))
                self.pred_df[self.target] = self.pred_df[self.target].apply(lambda x: literal_eval(str(x)))
            self.pred_df = self.pred_df.drop(self.target, axis=1).join(pd.DataFrame(self.mlb.transform(
                self.pred_df[self.target]), columns=self.mlb.classes_, index=self.pred_df.index))
            self.actual_value = self.pred_df[self.mlb.classes_]
            self.pred_df.drop(self.mlb.classes_, axis=1, inplace=True)
        else:
            self.actual_value = self.target_encoder.transform(self.pred_df[self.target])
            self.pred_df.drop(self.target, axis=1, inplace=True)

        df_dummies = pd.DataFrame()
        for i in self.categorical_variables:
            x = pd.get_dummies(self.pred_df[i], prefix=i, drop_first=True)
            df_dummies = pd.concat([df_dummies, x], axis=1)
            self.pred_df.drop(i, axis=1, inplace=True)
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "Categorical variables in pred_df data are transformed to dummy variables")
        self.pred_df = pd.concat([df_dummies, self.pred_df], axis=1)

        for col in list(self.pred_df.columns):
            if col not in self.final_features:
                self.pred_df.drop(col, axis=1, inplace=True)
                log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                            f"train data doesn't have one of the pred_df data columns {col}. "
                            "Dropping that columns from pred_df.")

        for col in self.final_features:
            if col not in list(self.pred_df.columns):
                self.pred_df[col] = 0
                log.VERBOSE(sys._getframe().f_lineno, __file__, __name__,
                            f"pred_df data doesn't have one of the train data columns {col}. "
                            "Creating that column with zero value in pred_df.")

        # check to assure the final features in train data and pred_df are same
        if len(self.final_features) != len(self.pred_df.columns):
            log.ERROR(sys._getframe().f_lineno, __file__, __name__,
                      "Train data and pred_df data columns are different.")
            return IXO_LOG_ERROR

    @method_header(
        description='''
            standardizing pred_df data.''')
    def standardize(self):
        # standardizing the data
        scalar = preprocessing.StandardScaler()
        self.pred_df = pd.DataFrame(scalar.fit_transform(self.pred_df), columns=self.pred_df.columns,
                                    index=self.pred_df.index)
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "pred_df data is standardized.")


if __name__ == '__main__':
    pass
