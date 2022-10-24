from meghnad.core.cust.rfm.cfg.config import RfmAnalyzerConfig
from utils.log import Log
from utils.common_defs import class_header, method_header

import itertools
import pandas as pd

log = Log()

# Merges RFM dataframe with columns like recency, frequency, monetary etc
def _merge_dataframes(rfm_def, rfm_group_details, col_name):
    rfm_def = pd.merge(rfm_def, rfm_group_details, how= 'left', left_on = col_name, right_on = 'codes')
    rfm_def.drop(columns = ['codes', col_name], inplace=True)
    rfm_def = rfm_def.rename(columns = {'description': col_name})

    return rfm_def

# Create RFM combinations
def _get_rfm_parameters(rfm_combinations):
    rfm_def = pd.DataFrame()
    for comb in rfm_combinations:
        tmp = pd.DataFrame({'rfm_group': '-'.join(comb),
                            'recency': comb[0],
                            'frequency':comb[1],
                            'monetary':comb[2]}, index = [0])

        rfm_def = rfm_def.append(tmp)


    rfm_def.set_index('rfm_group', inplace = True)

    return rfm_def

# Create customer groups based on given config parameters
def _get_customer_group(rfm_def:object):
    for index, row in rfm_def.iterrows():
        rec = row['recency']
        freq = row['frequency']
        mon = row['monetary']

        if (rec == 'R1') & ((freq == 'F1') | (freq == 'F2')):
            rfm_def.at[index, 'customer_group'] = 'Loyal_regular_consumers'
        elif (rec == 'R1') & (freq == 'F3'):
            rfm_def.at[index, 'customer_group'] = 'Recent_One_timers'
        elif (rec == 'R2') & (freq == 'F3'):
            rfm_def.at[index, 'customer_group'] = 'One_timers_at_risk'
        elif ((rec == 'R2') | (rec == 'R3')) & ((freq == 'F1') | (freq == 'F2')):
            rfm_def.at[index, 'customer_group'] = 'Ex-Loyal_regular_consumers'
        elif (rec == 'R3') & (freq == 'F3'):
            rfm_def.at[index, 'customer_group'] = 'One-timers_Churners'

    rfm_def = rfm_def.reset_index()

    return rfm_def

# Write RFM data to the path
def _write_to_datapath(rfm_def:object, path:str):
    rfm_def.to_csv(path, index = False)

# This function does a basic sanity check of RFM data which is modified by user
def _rfm_data_check(rfm_new:object, rfm_original:object):
    rec_comp = rfm_original.recency.compare(rfm_new.recency)
    freq_comp = rfm_original.frequency.compare(rfm_new.frequency)
    mon_comp = rfm_original.monetary.compare(rfm_new.monetary)
        
    flag = False
    if ((len(rec_comp) == 0) & (len(freq_comp) == 0) & (len(mon_comp) == 0)):
        flag = True

    return flag

class RfmDefinition():
    def __init__(self, *args, **kwargs):
        self.configs = RfmAnalyzerConfig()

    @method_header(
    description='''
    Prepares RFM definitions data and writes it in a given path''',
    arguments='''
    configs: config parameters
    rfm_def_path: path where RFM table will be stored''')
    def prepare_rfm_definition(configs:object, rfm_def_path:str):
        config = configs.get_rfm_configs()
        a = [config['recency_labels'], config['frequency_labels'], config['monetary_labels']] 
        rfm_combinations = list(itertools.product(*a))

        rfm_group_details = configs.get_rfm_group_details()
        rfm_group_details = pd.DataFrame(rfm_group_details.items(), columns = ['codes', 'description'])

        rfm_def = _get_rfm_parameters(rfm_combinations)
        rfm_def = _get_customer_group(rfm_def)
        

        rfm_def = _merge_dataframes(rfm_def, rfm_group_details, 'recency')
        rfm_def = _merge_dataframes(rfm_def, rfm_group_details, 'frequency')
        rfm_def = _merge_dataframes(rfm_def, rfm_group_details, 'monetary')

        rfm_def['customer_group'] = rfm_def['customer_group'] + "_" + rfm_def['monetary']

        _write_to_datapath(rfm_def, rfm_def_path)

    @method_header(
    description='''
    Get RFM definitions data from given path''',
    arguments='''
    rfm_def_path: path where RFM definition data is stored''',
    returns='''
    dataframe containing RFM definitions data''')
    def get_rfm_definition(self, rfm_def_path:str)->object:
        rfm_df = pd.read_csv(rfm_def_path)
        return rfm_df

    @method_header(
    description='''
    Set RFM definitions data to given path if and only if it passes basic sanity check''',
    arguments='''
    rfm_new:RFM dataframe that has been changed by user
    rfm_original:RFM dataframe that is generated based on config parameters
    rfm_def_path: path where RFM definition data is stored''',
    returns='''
    dataframe containing RFM definitions data''')
    def set_rfm_definition(self, rfm_new:object, rfm_original:object, rfm_def_path:str)->None:
        flag = _rfm_data_check(rfm_new, rfm_original)

        if flag == True:
            _write_to_datapath(rfm_new, rfm_def_path)
        else:
            log.VERBOSE(sys._getframe().f_lineno,
                        __file__, __name__,
                        "Please change only customer_group column and keep the other columns same.")
        

if __name__ == '__main__':
     pass



