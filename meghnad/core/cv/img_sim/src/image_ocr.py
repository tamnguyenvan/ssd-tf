#######################################################################################################################
# Image Text Extraction Module
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Sreyasha Sengupta
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from connectors.interfaces.interface import *
from meghnad.core.cv.img_sim.cfg.config import ImageSimConfig
from paddleocr import PaddleOCR

log = Log()

@class_header(
description='''
Text Extraction between two Images.''')
class TextExtraction():
    def __init__(self, *args, **kwargs):
        self.configs = ImageSimConfig()
        self.connector = {}

    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}

    @method_header(
    description='''
    Extract text''',
    arguments='''
    image: image input''',
    returns='''
    list of strings found in image''')
    def pred(self, image:object) -> [str]:
        texts_found = []
        config = self.configs.get_img_ocr_configs()
        if IXO_MEGHNAD_LOG_LEVEL < IXO_LOG_VERBOSE:
            show_log = False
        else:
            show_log = True

        try:
            ocr = PaddleOCR(use_angle_cls=True, lang=config['lang'], use_gpu=False, show_log=show_log)
            result = ocr.ocr(image, cls=True)
            for item in result:
                texts_found.append(item[1][0])
        except:
            pass

        return texts_found

if __name__ == '__main__':
    pass

