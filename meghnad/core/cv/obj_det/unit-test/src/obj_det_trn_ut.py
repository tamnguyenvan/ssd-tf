import unittest
from meghnad.core.cv.obj_det.src.tensorflow.train import TFObjDetTrn
from meghnad.core.cv.obj_det.src.tensorflow.inference import TFObjDetPred


def test_case1():
    """Training pipeline"""
    settings = ['light']
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    trainer = TFObjDetTrn(settings=settings)
    trainer.config_connectors(path)
    trainer.train(epochs=10)


def test_case2():
    """Test inference"""
    img_path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset\\images\\000a514fb1546570.jpg'
    predictor = TFObjDetPred(saved_dir='./checkpoints/best_saved_model')
    ret_value, (boxes, classes, scores) = predictor.predict(img_path)
    print(boxes.shape, classes.shape, scores.shape)


def _perform_tests():
    test_case2()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()