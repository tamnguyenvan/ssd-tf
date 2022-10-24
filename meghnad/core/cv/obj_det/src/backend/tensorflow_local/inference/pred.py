import os
import sys
import json

from PIL import Image, ImageDraw

from utils import ret_values
from utils.log import Log


log = Log()


class ModelInference:
    def __init__(self, model=None, prediction_postprocessing=None, output_dir='D://results'):
        self.model = model
        self.prediction_postprocessing = prediction_postprocessing
        self.output_dir = output_dir

    def predict(self, input_image_batch, history=None):
        if not self.model:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Model is null")
            return ret_values.IXO_RET_INVALID_INPUTS
        if not history:
            log.ERROR(sys._getframe().f_lineno,
                      __file__, __name__, "Model is not  yet fitted ")
            return ret_values.IXO_RET_INVALID_INPUTS
        predictions = self.model.predict(input_image_batch)
        if self.prediction_postprocessing:
            predictions = self.prediction_postprocessing(predictions)

        return ret_values.IXO_RET_SUCCESS, predictions

    def write_prediction(self, path, predictions):
        test_ann_file = os.path.join(path, 'test_annotations.json')
        with open(test_ann_file, 'r') as f:
            annotations = json.load(f)

        for i, val in enumerate(annotations):
            image_path = '%012d.jpg' % (val['image_id'])
            full_path = path + 'test\\' + image_path
            pred = predictions[i]
            shape = ((pred[0], pred[1]),
                     (pred[0] + pred[2], pred[1] + pred[3]))
            image = Image.open(full_path)
            draw = ImageDraw.Draw(image)
            draw.rectangle(shape, outline='red')
            if not (os.path.exists(self.output_dir)):
                os.mkdir(self.output_dir)
            image.save(self.output_dir + "\\" + image_path)
