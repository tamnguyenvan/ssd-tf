import sys

from utils import ret_values
from utils.log import Log

log = Log()


class ModelEvaluator:
    def __init__(self, model=None, test_dataset=None):
        self.model = model
        self.test_dataset = test_dataset

    def eval(self):
        if self.model is None:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Model is not fitted yet")
            return ret_values.IXO_RET_INVALID_INPUTS
        if len(self.test_dataset.cardinality().shape) == 0:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Test data is empty")
            return ret_values.IXO_RET_INVALID_INPUTS
        loss, metrics = self.model.evaluate(self.test_dataset)
        return ret_values.IXO_RET_SUCCESS, loss, metrics
