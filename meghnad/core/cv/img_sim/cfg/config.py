#######################################################################################################################
# Image Similarity Configuration 
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Sreyasha Sengupta
#######################################################################################################################
from utils.common_defs import *

_img_sim_cfg =\
{
    'stride_number': 3,
    'list_of_stride_combinations' :[[1,2,4,5],[2,3,5,6],[4,5,7,8],[5,6,8,9]],
    'repo_id': 'google/vit-base-patch16-224-in21k',
}

_img_ocr_cfg=\
{
    'lang': 'en'
}

class ImageSimConfig():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_img_sim_configs(self):
        return _img_sim_cfg.copy()

    def get_img_ocr_configs(self):
        return _img_ocr_cfg.copy()

if __name__ == '__main__':
    pass

