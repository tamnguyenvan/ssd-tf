from meghnad.core.cust.rfm.cfg.config import RfmAnalyzerConfig
from meghnad.core.cust.rfm.src.rfm_definition import RfmDefinition
from utils.common_defs import class_header, method_header
from utils.log import Log
from utils.ret_values import *

import os
import numpy as np
import pandas as pd
import datetime as dt

log = Log()

# Calculates recency column
def _get_recency_column(row:object, recency_groups:object, recency_labels:object) -> str: 
    if row['date_diff'] <= recency_groups[0]:
        return recency_labels[0]
    elif (recency_groups[0] < row['date_diff']) & (row['date_diff'] <= recency_groups[1]):
        return recency_labels[1]
    else: 
        return recency_labels[2]

 # Calculates frequency column
def _get_frequency_column(row:object, frequency_groups:object, frequency_labels:object) -> str:
    if row['order_cnt'] < frequency_groups[0] :
        return frequency_labels[0] 
    elif (row['order_cnt'] >= frequency_groups[0]) & (row['order_cnt'] < frequency_groups[1]):
        return frequency_labels[1] 
    else:
        return frequency_labels[2]

# Calculates monetary column
def _get_monetary_column(row:object, q1:float, q2:float, monetary_labels:object) -> str:
    if row['aov'] > q1:
        return monetary_labels[0] 
    elif (q1 >= row['aov']) & (row['aov'] > q2):
        return monetary_labels[1]
    else:
        return monetary_labels[2]

# This function assigns group to the customers who has ordered beyond the no_of_days
def _calculate_rfm_beyond_time_range(sales_order:object, date:object, no_of_days) -> object:
    rfm_df_old = sales_order.groupby(['cust_uid']).agg(last_order_date=('date_formatted',np.max))
    rfm_df_old[['order_cnt','revenue','aov']] = 0
    group_list = ['customer_group']

    for i in group_list:
        rfm_df_old['{}'.format(i)] = i + '_gt_'+str(no_of_days)+'days' 

    rfm_df_old = pd.DataFrame(rfm_df_old.to_records())
    rfm_df_old['selected_date'] = date 
    rfm_df_old['selected_date'] = rfm_df_old['selected_date'].astype('datetime64[D]',copy = False)
    rfm_df_old['date_diff'] = (rfm_df_old['selected_date'] - rfm_df_old['last_order_date']).dt.days

    rfm_df_old = rfm_df_old[rfm_df_old['date_diff'] > no_of_days]

    return rfm_df_old



@class_header(
description='''
RFM analyzer for Customers.''')
class RfmAnalyzer():
    def __init__(self, *args, **kwargs):
        self.configs = RfmAnalyzerConfig()
        self.rfm_def = RfmDefinition()
        self.connector = {}
        self.log = Log()

    @method_header(
    description='''
    Helper for configuring data connectors.''',
    arguments='''
    data_path: location of the sales data 
    data_type: type of the sales data ('csv' / 'json' / 'txt' etc.); currently only csv is supported
    feature_cols [optional]: attribute names in the data to be used as features during analysis
    target_cols [optional]: attribute names in the data to be used as output
    rfm_def_path: location of RFM definitions data 
    dir_to_save_result [optional]: location where the result to be saved''')
    def config_connectors(self, data_path:str, data_type:str, 
                          feature_cols:[str]=[], target_cols:[str]=[],
                          rfm_def_path:str=None,
                          dir_to_save_result:str=None,
                          *args, **kwargs):

        self.connector['data_path'] = data_path
        if data_type == 'csv':
            self.connector['data_type'] = data_type
        else:
            log.Error(sys._getframe().f_lineno,
                        __file__, __name__,
                        "Only csv is supported for the time being")

        self.connector['feature_cols'] = feature_cols
        self.connector['rfm_def_path'] = rfm_def_path

        if dir_to_save_result:
            self.connector['dir_to_save_result'] = dir_to_save_result
        else:
            self.connector['dir_to_save_result'] = self.configs.get_meghnad_configs('INT_PATH') + 'rfm/'
            if os.path.exists(self.connector['dir_to_save_result']):
                shutil.rmtree(self.connector['dir_to_save_result'])
            os.mkdir(self.connector['dir_to_save_result'])

    @method_header(
    description='''
    RFM analysis''',
    arguments='''
    base_data : sales order data
    def_df : rfm definitions to be used
    date : maximum date of sales order data''',
    returns='''
    dataframe containing different customer groups based on RFM definitions''')
    def rfm_analysis(self) -> object: 

        try:
            config = self.configs.get_rfm_configs()

            base_data = pd.read_csv(self.connector['data_path'])
            def_df = pd.read_csv(self.connector['rfm_def_path'])

            base_data = base_data[self.connector['feature_cols']]

            base_data['date_formatted'] = pd.to_datetime(base_data['date_formatted'])
            date = max(base_data.date_formatted)
        
            sales_order = base_data[(base_data['date_formatted'] <= date) & 
                                    (base_data['date_formatted'] >= (date - dt.timedelta(days = config['no_of_days'])))] 

            rfm_df = sales_order.groupby(['cust_uid']).agg(last_order_date = ('date_formatted',np.max),
                                                           order_cnt = ('order_id',np.size),
                                                           revenue = ('grand_total',np.sum))

            rfm_df['aov'] = rfm_df['revenue']/rfm_df['order_cnt']
            rfm_df = pd.DataFrame(rfm_df.to_records())
            rfm_df['selected_date'] = date 
            rfm_df['selected_date'] = rfm_df['selected_date'].astype('datetime64[D]', copy = False)
            rfm_df['date_diff'] = (rfm_df['selected_date'] - rfm_df['last_order_date']).dt.days

            ## Recency Column
            rfm_df['rec'] = rfm_df.apply(lambda x:_get_recency_column(x, config['recency_groups'], config['recency_labels']),axis=1)

            ## Frequency Column
            rfm_df['freq'] = rfm_df.apply(lambda x: _get_frequency_column(x, config['frequency_groups'], config['frequency_labels']),axis=1)

            ## Monetary Column
            q1 = rfm_df['aov'].quantile(q=config['qa'])
            q2 = rfm_df['aov'].quantile(q=config['qb'])

            rfm_df['mon'] = rfm_df.apply(lambda x: _get_monetary_column(x, q1, q2, config['monetary_labels']), axis=1)

            ## Total RFM groups Column  
            rfm_df['rfm_group'] = rfm_df.apply(lambda x:'%s-%s-%s' % (x['rec'],x['freq'],x['mon']),axis=1)

            ## Final RFM groups Column  
            rfm_df = pd.merge(rfm_df,def_df[['rfm_group','customer_group']], on = ['rfm_group'],how = 'left') 

            ## for rfm_df_old
            rfm_df = rfm_df.append(_calculate_rfm_beyond_time_range(sales_order, date, config['no_of_days']))

            rfm_df.drop(columns=["rec", "freq", "mon", "rfm_group"], inplace = True)

        except:
            return IXO_RET_GENERIC_FAILURE

        return rfm_df, IXO_RET_SUCCESS

if __name__ == '__main__':
     pass


 
