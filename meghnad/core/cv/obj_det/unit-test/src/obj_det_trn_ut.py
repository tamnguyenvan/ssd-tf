import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus, 'GPU')

from meghnad.core.cv.obj_det.src.tensorflow.train import TFObjDetTrn
from meghnad.core.cv.obj_det.src.tensorflow.inference import TFObjDetPred
import unittest


def test_case1():
    """Test training pipeline"""
    settings = ['light']
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    trainer = TFObjDetTrn(settings=settings)
    trainer.config_connectors(path)
    trainer.train(epochs=10)


def test_case2():
    """Test inference"""
    img_path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset\\test.jpg'
    predictor = TFObjDetPred(saved_dir='./checkpoints')
    ret_value, (boxes, classes, scores) = predictor.predict(img_path)
    print(boxes.shape, classes.shape, scores.shape)


def _perform_tests():
    test_case1()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
