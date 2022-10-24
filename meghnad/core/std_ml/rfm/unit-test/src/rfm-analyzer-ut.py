from meghnad.cfg.config import MeghnadConfig
from meghnad.core.cust.rfm.src.rfm_definition import RfmDefinition
from meghnad.core.cust.rfm.src.rfm_analyzer import RfmAnalyzer
from utils.ret_values import *
from utils.log import Log

import json
import os, gc

log = Log()

def _cleanup():
    gc.collect()

def _write_results(result, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    result.to_csv(results_path + "tc_rfm_result.csv", index = False)

def _tc_1(rfm_analyzer, testcases_path, results_path):
    data_path = testcases_path + "UK_Ploom_sales_order.csv"
    data_type = 'csv'
    feature_cols = ['order_id',	'grand_total',	'cust_uid', 'date_formatted']
    rfm_def_path = testcases_path + 'rfm_def.csv'
    dir_to_save_result = results_path

    rfm_analyzer.config_connectors(data_path=data_path, data_type=data_type,
                            feature_cols=feature_cols, rfm_def_path=rfm_def_path,  dir_to_save_result = dir_to_save_result)

    rfm_result, ret_val = rfm_analyzer.rfm_analysis()

    if ret_val == IXO_RET_SUCCESS:
        results_path += "tc_1/"
        _write_results(rfm_result, results_path)
    elif ret_val == IXO_RET_GENERIC_FAILURE:
        log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      "Error in RFM Analyzer")

def _tc_2():
    rfm_def_path = 'C:\\Users\\Souvik\\source\\repos\\rfm_2022_05_16\\ixolerator\\meghnad\\core\\cust\\rfm\\unit-test\\testcases\\rfm_def.csv'
    rfm_def_obj = RfmDefinition()
    rfm_original = rfm_def_obj.get_rfm_definition(rfm_def_path)
    rfm_def = rfm_original.copy()

    rfm_def.set_index("rfm_group", inplace = True)
    
    #rfm_def.at["R1-F2-M3",'customer_group']='test'
    #print(rfm_def)

    rfm_def.at["R1-F2-M3",'monetary']='test'
    print(rfm_def)

    rfm_def = rfm_def.reset_index()

    rfm_def_obj.set_rfm_definition(rfm_def, rfm_original,rfm_def_path)


    

def _perform_tests():
    rfm_analyzer = RfmAnalyzer()

    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/cust/rfm/unit-test/"
    testcases_path = ut_path + "testcases/"
    results_path = ut_path + "results/"

    _tc_1(rfm_analyzer, testcases_path, results_path)

    #_tc_2()

if __name__ == '__main__':
    _perform_tests()

    _cleanup()

