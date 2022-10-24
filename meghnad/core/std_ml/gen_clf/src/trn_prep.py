#######################################################################################################################
# Preprocessing of train data for Generic Classifier.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kirankumar A M
#######################################################################################################################

# Import Libraries
from utils.log import Log
from utils.common_defs import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.std_ml.gen_clf.cfg.config import GenericClfConfig
import pandas as pd
import numpy as np
import sys, os
from ast import literal_eval

# pre-processing modules
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

# statistics modules
import scipy.stats as stats

log = Log()


@class_header(
    description='''
    Generic classifier training data preprocessing pipeline.''')
class GenericClfTrnPrep():
    def __init__(self, connector: dict):
        self.connector = connector
        self.target = self.connector['target_cols']
        self.multi_labels = self.connector['multi_labels']
        self.configs = GenericClfConfig(MeghnadConfig)
        self.drop = list()

    @method_header(
        description='''
            Pre-processing train data.''',
        returns='''
            a 8 member tuple having the pre-processed train data, final target_list, features dropped from train data, 
            initial features list, list of categorical variables, initial target column, encoders for target variable.
            ''')
    def prep(self) -> (object, [str], [str], [str], [str], str, object, object):

        # GenericClfPrep.read_data(self)
        self.read_data()

        self.mlb = None

        self.encoder = None

        if not self.connector['multi_labels']:
            self.label_encode_target_variable()

        self.drop_rows_and_columns()

        self.numerical_categorical()

        if self.connector['multi_labels']:
            self.cleaning_target_variable()
            self.label_dummy_encoding()
        else:
            self.target_list = self.connector['target_cols']
            self.label_dummy_encoding()

        if not self.connector['multi_labels']:
            self.imbalance_correction()

        self.standardizing()

        return self.train, self.target_list, self.drop, self.features_list, self.categorical_variables, self.target, \
               self.mlb, self.encoder

    @method_header(
        description='''
            Reading train data.''')
    def read_data(self):
        if self.connector['data_org'] in ['multi_dir_general', 'multi_dir_periodic']:
            self.train = pd.DataFrame()
            for j, folder in enumerate(os.listdir(self.connector['data_path'])):
                df_train = pd.DataFrame()
                dire = self.connector['data_path'] + '/' + folder
                i = 0
                for file in os.listdir(dire):
                    if i == 0 and j == 0:
                        if self.connector['data_type'] in ['csv', 'txt']:
                            df = pd.read_csv(dire + '/' + file, sep=self.connector['seperator'])
                        elif self.connector['data_type'] == 'excel':
                            df = pd.read_excel(dire + '/' + file)
                        df_train = df.copy()
                        i += 1
                    else:
                        if self.connector['data_type'] in ['csv', 'txt']:
                            df = pd.read_csv(dire + '/' + file)
                        elif self.connector['data_type'] == 'excel':
                            df = pd.read_excel(dire + '/' + file)
                        df_train = df_train.append(df, ignore_index=True)
                        i += 1
                data = df_train.copy()
                if self.connector['data_org'] == 'multi_dir_general':
                    data[self.connector['target_cols']] = folder
                self.train = self.train.append(data, ignore_index=True)
            if not self.connector['feature_cols'] is None:
                self.train = pd.concat((self.train[self.connector['feature_cols']],
                                        self.train[self.connector['target_cols']]), axis=1)
            log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                       "Training data from multiple directories are read and stored in train dataframe")
        else:
            if self.connector['data_type'] in ['csv', 'txt']:
                self.train = pd.read_csv(self.connector['data_path'], sep=self.connector['seperator'],
                                         usecols=None if self.connector['feature_cols'] is None
                                         else self.connector['feature_cols']+[self.connector['target_cols']])
            elif self.connector['data_type'] == 'excel':
                self.train = pd.read_excel(self.connector['data_path'], usecols=None
                                           if self.connector['feature_cols'] is None
                                           else self.connector['feature_cols']+[self.connector['target_cols']])
            log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                       "Training data is read and stored in train dataframe")

        self.features_list = [x for x in self.train.columns if x != self.target]

    @method_header(
        description='''
            Label encoding the target variable.''')
    def label_encode_target_variable(self):
        self.encoder = preprocessing.LabelEncoder()
        if (self.train[self.target].dtypes == 'O') or (self.train[self.target].dtypes == 'bool'):
            self.train[self.target] = self.encoder.fit_transform(self.train[self.target])
            log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                       "Target variable is label encoded")

    @method_header(
        description='''
            Drop rows and columns based on certain conditions.''')
    def drop_rows_and_columns(self):
        # replacing empty strings and blank spaces in data to Nan
        self.train.replace('', np.nan, inplace=True)
        self.train.replace(' ', np.nan, inplace=True)
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "Replacing empty strings and blank spaces in data to Nan")

        # convert those empty string replaced columns from object dtype to float dtype based on other values
        for i in self.train.columns:
            if (self.train[i].dtype == 'O') and (self.train[i].isna().sum() > 0):
                try:
                    self.train[i] = self.train[i].astype(float)
                    log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                               "Empty string replaced to Nan columns are changed from object to float data type")
                except ValueError:
                    continue

        # dropping columns with more than half of the rows having missing values
        drop_1 = self.train.isna().sum().index[self.train.isna().sum() > 0.5 * len(self.train)]
        self.train.drop(drop_1, axis=1, inplace=True)
        if len(drop_1) > 0:
            for i in drop_1:
                self.drop.append(i)
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "Dropping columns having missing values in more than half of the rows in "
                   "train data")

        # dropping rows with missing values
        self.train.dropna(axis=0, inplace=True)
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "Dropping rows with missing values in train data")

        # reset index
        self.train = self.train.reset_index(drop=True)

        # drop nominal features like name, ID, S.No.
        # if the number of unique values of a column is equal to the length of the dataframe, that column is dropped
        drop_2 = self.train.nunique().index[self.train.nunique() == len(self.train)]
        self.train.drop(drop_2, axis=1, inplace=True)
        if len(drop_2) > 0:
            for i in drop_2:
                self.drop.append(i)
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "Dropping columns with no. of unique values equal to the length of data (aka nominal variables)")

        # drop columns with the same value in all rows
        drop_3 = list(self.train.nunique().index[self.train.nunique() == 1])
        self.train.drop(drop_3, axis=1, inplace=True)
        if len(drop_3) > 0:
            for i in drop_3:
                self.drop.append(i)
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "Dropping columns with same value in all rows")

    @method_header(
        description='''
            Get numerical and categorical variables separately after checking its importance.''')
    def numerical_categorical(self):
        # Getting the categorical variables (excluding target variable) in a list
        obj_cat = self.train.dtypes.index[self.train.dtypes == 'object'].values
        num_cat = self.train.dtypes.index[self.train.dtypes != 'object'].values
        object_cat_variables = []
        num_cat_variables = []

        for i in obj_cat:
            object_cat_variables.append(i)

        for i in num_cat:
            if (i not in self.target) and (self.train[i].nunique() <= 5):
                num_cat_variables.append(i)

        # categorical_variables
        self.categorical_variables = object_cat_variables + num_cat_variables
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "Categorical variables are retrieved from data")

        # getting the numerical variables in a separate list
        self.numerical_variables = list(set(self.train.columns) - set(self.categorical_variables))
        self.numerical_variables = [i for i in self.numerical_variables if i != self.target]
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "Numerical variables are retrieved from data")

        if not self.multi_labels:

            # Tests for dependency on categorical variables (Chi-square Test)
            # H0: Feature is independent on the target variable
            # H1: Feature is dependent on the target variable
            for i in self.categorical_variables:
                arr = np.array(pd.crosstab(self.train[i], columns=self.train[self.target]))
                chi_sq_stat, p_value, deg_freedom, exp_freq = stats.chi2_contingency(arr, correction=False)
                if p_value > 0.05:
                    self.train.drop(i, axis=1, inplace=True)
                    self.drop.append(i)
                    self.categorical_variables.remove(i)
                    log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                               f"Categorical variable {i} is dropped from data as it is independent"
                               " of target variable")

            # Tests for dependency on numerical variables (Kruskal-Wallis H-test)
            # The Kruskal-Wallis H-test tests the null hypothesis that the population
            # median of all the groups are equal.  It is a non-parametric version of ANOVA.
            # H0: No differences among groups in the feature
            # H1: There is difference among groups in the feature
            for i in self.numerical_variables:
                stat, p_value = stats.kruskal(self.train.loc[self.train[self.target] == 1, i],
                                              self.train.loc[self.train[self.target] == 0, i])
                if p_value > 0.05:
                    self.train.drop(i, axis=1, inplace=True)
                    self.drop.append(i)
                    self.numerical_variables.remove(i)
                    log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                               f"Numerical variable {i} is dropped from data as its median is same across all classes")

    @method_header(
        description='''
            Cleaning and Transforming the multi-label target variable.''')
    def cleaning_target_variable(self):

        self.mlb = preprocessing.MultiLabelBinarizer()

        try:
            self.train[self.target] = self.train[self.target].apply(lambda x: literal_eval(str(x)))
        except:
            self.train[self.target] = self.train[self.target].apply(lambda x: x.split(self.connector['multi_labels_'
                                                                                                     'seperator']))
            self.train[self.target] = self.train[self.target].apply(lambda x: literal_eval(str(x)))

        self.train = self.train.drop(self.target, axis=1).join(
            pd.DataFrame(self.mlb.fit_transform(self.train[self.target]),
                         columns=self.mlb.classes_, index=self.train.index))

        self.categorical_variables.remove(self.target)

        self.target_list = self.mlb.classes_

        log.STATUS(sys._getframe().f_lineno, __file__, __name__, "Target column is transformed to binary matrix")

    @method_header(
        description='''
            Label encoding and creation of dummy variables for train data.''')
    def label_dummy_encoding(self):
        train_dummies = pd.DataFrame()
        for i in self.categorical_variables:
            x = pd.get_dummies(self.train[i], prefix=i, drop_first=True)
            train_dummies = pd.concat([train_dummies, x], axis=1)
            self.train.drop(i, axis=1, inplace=True)
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "Categorical variables in train data are transformed to dummy variables")

        self.train = pd.concat([train_dummies, self.train], axis=1)
        # final features
        self.features = [x for x in self.train.columns if x not in self.target_list]

    @method_header(
        description='''
            performing imbalance correction (SMOTE) in train data.''')
    def imbalance_correction(self):
        imb = pd.crosstab(self.train[self.target_list], columns='count') / self.train.shape[0]
        imb = imb.values.flatten()
        if sum(imb < 1/self.train[self.target_list].nunique()):
            x_resampled, y_resampled = SMOTE().fit_resample(self.train[self.features], self.train[self.target_list])
            self.train = pd.concat([x_resampled, y_resampled], axis=1)
            log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                       "SMOTE imbalance correction done on train data.")

    @method_header(
        description='''
            Standardizing the train data.''')
    def standardizing(self):
        scalar = preprocessing.StandardScaler()
        y_train = self.train[self.target_list]
        train_std = pd.DataFrame(scalar.fit_transform(self.train[self.features]))
        train_std.columns = self.features
        self.train = pd.concat([train_std, y_train], axis=1)
        log.STATUS(sys._getframe().f_lineno, __file__, __name__,
                   "Train data is standardized")


if __name__ == '__main__':
    pass
