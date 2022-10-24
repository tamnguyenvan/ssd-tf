#######################################################################################################################
# Image Similarity Module
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Sreyasha Sengupta
#######################################################################################################################

from utils.ret_values import *
from meghnad.cfg.config import MeghnadConfig
from meghnad.core.cv.img_sim.src.image_similarity import ImageSimilarity
from meghnad.core.cv.img_sim.src.image_ocr import TextExtraction

import json, gc

import unittest

def _cleanup():
    gc.collect()

def _write_results(result, results_path):
    with open(results_path, 'w') as file:
        file.write(json.dumps(result)) 

def _tc_1(img_sim, testcases_path, results_path):
    base_img = testcases_path + "z1.jpg"
    images = [testcases_path +'z2.jpg', testcases_path +'z3.jpg']
    img_sim_result_all = {}
    file_name = results_path + 'results_tc_1.txt'

    for image in images:
        ret_val, img_sim_result = img_sim.pred(base_img, image)
        img_sim_result_all[image] = img_sim_result

    _write_results(img_sim_result_all, file_name)

def _tc_2(text_extractor, testcases_path, results_path):
    image = testcases_path +'z2.jpg'
    file_name = results_path + 'results_tc_2.txt'

    result = text_extractor.pred(image)

    _write_results(result, file_name)

def _perform_tests():
    img_sim = ImageSimilarity()

    ut_path = MeghnadConfig().get_meghnad_configs('BASE_DIR') + "core/cv/img_sim/unit-test/"
    testcases_path = ut_path + "testcases/"
    results_path = ut_path + "results/"

    _tc_1(img_sim, testcases_path, results_path)

    text_extractor = TextExtraction()

    _tc_2(text_extractor, testcases_path, results_path)

if __name__ == '__main__':
    _perform_tests()

    unittest.main()

    _cleanup()

