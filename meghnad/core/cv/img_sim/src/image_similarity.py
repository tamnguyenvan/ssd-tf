#######################################################################################################################
# Image Similarity Module
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Sreyasha Sengupta
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from utils.log import Log
from meghnad.core.cv.img_sim.cfg.config import ImageSimConfig
from meghnad.core.cv.img_sim.src.image_ocr import TextExtraction

import cv2, torch
from transformers import ViTFeatureExtractor

from fuzzywuzzy import fuzz
import sys

log = Log()

def _cosine_similarity_scores(features1:object, features2:object,
                              list_of_stride_combinations:[[int]], full_image:bool=False) -> dict:
    """
    This function calculates the cosine similarity between 2 sets of features of images.
    Arguments:
    features1: Tensor representing features of image 1
    features2: Tensor representing features of image 2
    full_image: Boolean variable signifying if the feature is of full image or of strides
    """
    similarity_scores = {}

    if not full_image:
        count = 0
        for comparison_list in list_of_stride_combinations:
            stride_list = []
            count += 1
            for i in comparison_list:
                for j in comparison_list:                
                    cos = torch.nn.CosineSimilarity(dim=0)
                    output = cos(features1[i], features2[j])
                    stride_list.append(output)                
            stride_name = "stride_sim_score_" + str(count)
            similarity_scores[stride_name] = max(stride_list).numpy().tolist()
    else:
        cos = torch.nn.CosineSimilarity(dim=0)
        similarity_scores['full_sim_score'] = cos(features1['full_img_feature'], features2['full_img_feature'])
        similarity_scores['full_sim_score'] = similarity_scores['full_sim_score'].numpy().tolist()

    return similarity_scores

def _fuzzy_scores(text_list_1:[str], text_list_2:[str]) -> list:
    """
    This function calculates the similarty score between two text inputs.
    Arguments:
    text_list_1: Strings representing first text
    text_list_2: Strings representing second text
    """
    scores_ratio = []
    scores_partial_ratio = []
    scores_token_sort_ratio = []
    scores_token_set_ratio = []

    for i in text_list_1:
        rlist = {}
        prlist = {}
        tsrlist = {}
        tsetlist = {}

        for j in text_list_2:
            ratio = fuzz.ratio(i,j)
            partial_ratio = fuzz.partial_ratio(i,j)
            token_sort_ratio = fuzz.token_sort_ratio(i,j)
            token_set_ratio = fuzz.token_set_ratio(i,j)

            rlist[str(i+" and "+j)] = ratio
            prlist[str(i+" and "+j)] = partial_ratio
            tsrlist[str(i+" and "+j)] = token_sort_ratio
            tsetlist[str(i+" and "+j)] = token_set_ratio

        scores_ratio.append(max(zip(rlist.keys(), rlist.values())))
        scores_partial_ratio.append(max(zip(prlist.keys(), prlist.values())))
        scores_token_sort_ratio.append(max(zip(tsrlist.keys(), tsrlist.values())))
        scores_token_set_ratio.append(max(zip(tsetlist.keys(), tsetlist.values())))

    return scores_ratio, scores_partial_ratio, scores_token_sort_ratio, scores_token_set_ratio

@class_header(
description='''
Image Similarity between two images.''')
class ImageSimilarity():
    def __init__(self, *args, **kwargs):
        self.configs = ImageSimConfig()
        self.connector = {}
        self.text_extractor = TextExtraction()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.configs.get_img_sim_configs()['repo_id'])

    def config_connectors(self, *args, **kwargs):
        self.connector_pred = {}

    @method_header(
    description='''
    Image Similarity Score''',
    arguments='''
    img_path_1: location of the reference image
    img_path_1: location of the target image''',
    returns='''
    Final Similarity Scores between two images''')
    def pred(self, base_image_path:str, image_path:str) -> object:
        try:
            img_path_1 = base_image_path
            img_path_2 = image_path

            config = self.configs.get_img_sim_configs()
            final_similarity_scores = self._similarity_score(img_path_1, img_path_2,
                                                             config['stride_number'],
                                                             config['list_of_stride_combinations'])
            return IXO_RET_SUCCESS, final_similarity_scores
        except Exception as e:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__,
                      str(e,e.__traceback__.tb_lineno))
            return IXO_RET_GENERIC_FAILURE, None

    def _similarity_score(self, img_path_1: str, img_path_2: str,
                          stride_number: int, list_of_stride_combinations: [[int]]) -> dict:
        """
        This function takes two images and calculates the similarity scores between them
        Arguments:
        img_path_1: Path of the first image
        img_path_2: Path of the second image
        """
        image1 = cv2.imread(img_path_1)
        image2 = cv2.imread(img_path_2)

        stride_features1 = self._feature_matrix(image1, stride_number, stride=True)
        stride_features2 = self._feature_matrix(image2, stride_number, stride=True)
        full_features1 = self._feature_matrix(image1, stride_number, stride=False)
        full_features2 = self._feature_matrix(image2, stride_number, stride=False)

        #### COMMON FEATURES #####
        stride_similarity_scores = _cosine_similarity_scores(stride_features1, stride_features2,\
                                                             list_of_stride_combinations, full_image=False)
        full_similarity_score = _cosine_similarity_scores(full_features1, full_features2,\
                                                          list_of_stride_combinations, full_image=True)

        ####### OCR PIPELINE #######
        ocr_similarity_score = {}
        texts_in_image_1 = ' '.join(self.text_extractor.pred(img_path_1))
        texts_in_image_2 = ' '.join(self.text_extractor.pred(img_path_2))
        scores_ratio, scores_partial_ratio, scores_token_sort_ratio, scores_token_set_ratio =\
            _fuzzy_scores([texts_in_image_1], [texts_in_image_2])
        ocr_similarity_score['OCR_sim_score'] = scores_partial_ratio

        ### Final Dictionary ####
        final_similarity_scores = {**ocr_similarity_score, **full_similarity_score}
        final_similarity_scores = {**final_similarity_scores, **stride_similarity_scores}

        return final_similarity_scores

    def _feature_matrix(self, image:object, stride_number:int, stride:bool=False) -> dict:
        """
        This function extracts feature from an image
        Arguments:
        image: An matrix representing an Image
        stride: Boolean variable signifying if the image has to be divided into strides or not
        """
        feature_dict = {}

        if stride:
            count = 0
            for i in range(stride_number):
                for j in range(stride_number):
                    count += 1
                    name = (count)
                    start_x = int(i) * int(image.shape[1] / stride_number)
                    start_y = int(j) * int(image.shape[0] / stride_number)
                    end_x = int(image.shape[1] / stride_number) * (i + 1)
                    end_y = int(image.shape[0] / stride_number) * (j + 1)
                    crop = image[start_y:end_y, start_x: end_x]
                    features = self.feature_extractor(crop, return_tensors='pt')
                    features = features['pixel_values']
                    feature_dict[name] = features.reshape((-1))
        else:
            features = self.feature_extractor(image, return_tensors='pt')
            features = features['pixel_values']
            feature_dict["full_img_feature"] = features.reshape((-1))

        return feature_dict

if __name__ == '__main__':
    pass

