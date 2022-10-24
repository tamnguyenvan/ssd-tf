#######################################################################################################################
# Interface for NLP component in Meghnad.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.ret_values import *

from meghnad.metrics.metrics import ClfMetrics

from meghnad.core.cust.rfm.src.rfm_definition import RfmDefinition
from meghnad.core.cust.rfm.src.rfm_analyzer import RfmAnalyzer

from meghnad.core.std_ml.gen_clf.src.trn import GenericClfTrn
from meghnad.core.std_ml.gen_clf.src.pred import GenericClfPred
from meghnad.core.std_ml.gen_clf.src.pred_prep import GenericClfPredPrep

