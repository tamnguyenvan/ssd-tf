from utils.common_seed import *

import os
os.environ['PYTHONHASHSEED'] = str(IXO_SEED)

import random
random.seed(IXO_SEED)

