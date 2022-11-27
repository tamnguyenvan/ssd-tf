import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus, 'GPU')

from meghnad.core.cv.obj_det.src.tensorflow.train import TFObjDetTrn
from meghnad.core.cv.obj_det.src.tensorflow.data_loader import TFObjDetDataLoader
from meghnad.core.cv.obj_det.cfg import ObjDetConfig
import unittest


def test_case1():
    """Test training pipeline"""
    settings = ['light']
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    trainer = TFObjDetTrn(settings=settings)
    trainer.config_connectors(path)
    trainer.train(epochs=10)


def test_case2():
    """Test data loader"""
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    cfg_obj = ObjDetConfig()
    data_cfg = cfg_obj.get_data_cfg()
    model_cfg = cfg_obj.get_model_cfg('MobileNetV2')
    dataloader = TFObjDetDataLoader(path, data_cfg, model_cfg)

    for images, gt_confs, gt_locs in dataloader.train_dataset.take(1):
        break

    image = images.numpy()[0]
    gt_conf = gt_confs.numpy()[0]
    gt_loc = gt_locs.numpy()[0]


def _perform_tests():
    test_case1()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
