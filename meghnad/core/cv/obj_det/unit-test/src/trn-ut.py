import sys
sys.path.append('D:\\ixolerator')
import os


import glob
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus, 'GPU')

import cv2
from meghnad.core.cv.obj_det.src.tensorflow.data_loader.data_loader import *
from meghnad.core.cv.obj_det.src.tensorflow.model_loader.model_loader import *
from meghnad.core.cv.obj_det.src.tensorflow.train.train import *
from meghnad.core.cv.obj_det.src.tensorflow.train.eval import *
from meghnad.core.cv.obj_det.src.tensorflow.inference.pred import *
from meghnad.core.cv.obj_det.src.tensorflow.inference import vis_utils
from meghnad.core.cv.obj_det.src.tensorflow.model_loader.ssd.utils.ssd_box_utils import encode
from meghnad.core.cv.obj_det.src.tensorflow.model_loader.ssd.utils.ssd_loss_utils import SSDLoss
import meghnad.core.cv.obj_det.cfg.config as cfg
import unittest

from meghnad.core.cv.obj_det.src.tensorflow.model_loader.ssd.anchors import generate_default_boxes
from meghnad.core.cv.obj_det.src.tensorflow.model_loader.ssd.utils.ssd_box_utils import compute_target



def test_case1(path):
    config = cfg.ObjDetConfig()
    model_name = 'MobileNetV2'
    data_config = config.get_data_cfg()
    model_config = config.get_model_cfg(model_name)
    model_params = config.get_model_params(model_name)
    print('Model name:', model_name)
    print('Model config')
    print(model_config)
    print('Model params')
    print(model_params)

    img_size = model_config['input_shape'][:2]
    print('image size', img_size)
    d_loader = TfObjDetDataLoader(
        num_classes=data_config['num_classes'],
        batch_size=model_params['batch_size'],
        img_size=img_size,
        scales=model_config['scales'],
        feature_map_sizes=model_config['feature_map_sizes'],
        aspect_ratios=model_config['aspect_ratios']
    )
    d_loader.load_data_from_directory(
        path=path, augment=False, rescale=False, rand_flip=False, rotate=False
    )
    m_loader = TfObjDetModelLoader(
        aarch=model_name,
        num_classes=data_config['num_classes'],
        model_config=model_config
    )
    m_loader.load_model()
    trainer = TfObjDetTrn(
        data_loader=d_loader,
        model_loader=m_loader,
        model_config=model_config,
        learning_rate=model_params['learning_rate'],
        loss=SSDLoss(model_config['neg_ratio'], data_config['num_classes'])
    )
    trainer.compile_model()
    trainer.train(epochs=200)

    best_checkpoint_path = trainer.get_best_model()
    evaluator = TfObjDetEval(
        m_loader,
        model_config,
        d_loader,
        ckpt_path=best_checkpoint_path,
        phase='validation'
    )
    evaluator.eval()

def test_case2(path):
    config = cfg.ObjDetConfig()
    model_name = 'MobileNetV2'
    data_config = config.get_data_cfg()
    model_config = config.get_model_cfg(model_name)
    model_params = config.get_model_params(model_name)
    print('Model config')
    print(model_config)
    print('Model params')
    print(model_params)

    img_size = model_config['input_shape'][:2]
    print('image size', img_size)
    d_loader = TfObjDetDataLoader(
        num_classes=data_config['num_classes'],
        batch_size=model_params['batch_size'],
        img_size=img_size,
        scales=model_config['scales'],
        feature_map_sizes=model_config['feature_map_sizes'],
        aspect_ratios=model_config['aspect_ratios']
    )
    d_loader.load_data_from_directory(
        path=path, augment=False, rescale=False, rand_flip=False, rotate=False
    )
    m_loader = TfObjDetModelLoader(
        aarch=model_name,
        num_classes=data_config['num_classes'],
        model_config=model_config
    )
    m_loader.load_model()

    evaluator = TfObjDetTrn(
        m_loader,
        model_config,
        d_loader,
        ckpt_path='./checkpoints/MobileNetV2_best.ckpt',
        phase='test',
        image_out_dir='D:\\output',
        draw_predictions=True,
        from_hub = False
    )
    evaluator.eval()

def test_case3(dataset_path):
    image_dir = os.path.join(dataset_path, 'images')
    test_ann_file = os.path.join(dataset_path, 'test_annotations.json')
    import json
    from pycocotools.coco import COCO

    coco = COCO(test_ann_file)
    categories = coco.getCatIds()
    print(categories)
    img_ids = coco.getImgIds()
    check_img_ids = img_ids[:5]

    for i, img_id in enumerate(check_img_ids):
        img_info = coco.loadImgs([img_id])[0]
        filename = img_info['file_name']
        image_path = os.path.join(image_dir, filename)
        img = cv2.imread(image_path)

        ann_ids = coco.getImgIds([img_id])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            x, y, w, h = ann['bbox']
            cate_id = ann['category_id']
            cate_info = coco.loadCats([cate_id])[0]

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, cate_info['name'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imwrite(f'test{i+1}.png', img)

def _perform_tests():
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    test_case2(path)


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
