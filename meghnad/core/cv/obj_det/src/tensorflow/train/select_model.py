from meghnad.core.cv.obj_det.src.tensorflow.model_loader import ssd
from utils.common_defs import class_header, method_header


@class_header()
class TFObjDetSelectModel:
    def __init__(self, model_configs):
        self.model_configs = model_configs
        self.models = []
        for model_config in self.model_configs:
            model = ssd(
                model_config['arch'],
                model_config['input_shape'],
                model_config['num_classes'],
                model_config['num_anchors']
            )
            self.models.append(model)
