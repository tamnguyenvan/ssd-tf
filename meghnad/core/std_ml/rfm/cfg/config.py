from utils.common_defs import *
#from utils.ret_values import *
#from utils.log import Log
#from meghnad.cfg.config import MeghnadConfig

import sys

#log = Log()

rfm_cfg =\
{
    'no_of_days': 365,
    'recency_groups' : [60,180],  
    'recency_labels' : ['R1','R2','R3'],
    'frequency_groups' : [2,4],
    'frequency_labels' : ['F3','F2','F1'],
    'qa' : 0.8,
    'qb' : 0.5,
    'monetary_labels' : ['M1','M2','M3']
}

rfm_group_details = \
{
    'R1': 'Days_lte_' + str(rfm_cfg["recency_groups"][0]),
    'R2': 'Days_' + str(rfm_cfg["recency_groups"][0] + 1) + '_to_lte_' + str(rfm_cfg["recency_groups"][1]),
    'R3': 'Days_' + str(rfm_cfg["recency_groups"][1] + 1) + '_to_lte_' + str(rfm_cfg['no_of_days']),
    'F1': 'order_count_gte_' + str(rfm_cfg["frequency_groups"][1]),
    'F2': 'order_count_' + str(rfm_cfg["frequency_groups"][0]) + '_to_' + str(rfm_cfg["frequency_groups"][1] - 1),
    'F3': 'order_count_1',
    'M1': 'High',
    'M2': 'Medium',
    'M3': 'Low',
}


class RfmAnalyzerConfig():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_rfm_configs(self):
        return rfm_cfg.copy()

    def get_rfm_group_details(self):
        return rfm_group_details.copy()


if __name__ == '__main__':
    pass
