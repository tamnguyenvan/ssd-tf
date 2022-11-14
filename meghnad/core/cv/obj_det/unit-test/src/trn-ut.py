import sys
sys.path.append('D:\\10-27-22\ixolerator')
import os


import glob
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus, 'GPU')

import cv2
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.data_loader.data_loader import *
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.model_loader import *
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.train.train import *
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.train.eval import *
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.inference.pred import *
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.inference import vis_utils
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.models.ssd.utils.box_utils import encode
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.models.ssd.utils.loss_utils import SSDLoss
from meghnad.core.cv.obj_det.cfg import ObjDetConfig
import unittest

from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.models.ssd.anchors import generate_default_boxes
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.models.ssd.utils.box_utils import compute_target


def test_case9(path):
    config = ObjDetConfig()

    settings = ['light']
    trainer = ModelTrainer(settings=settings)
    trainer.train(epochs=10)

    # model_name = 'MobileNetV2'
    # data_config = config.get_data_cfg()
    # model_config = config.get_model_cfg(model_name)
    # model_params = config.get_model_params(model_name)
    # print('Model name:', model_name)
    # print('Model config')
    # print(model_config)
    # print('Model params')
    # print(model_params)

    # # label_encoder = LabelEncoder()
    # img_size = model_config['input_shape'][:2]
    # print('image size', img_size)
    # d_loader = DataLoader(
    #     num_classes=data_config['num_classes'],
    #     batch_size=model_params['batch_size'],
    #     img_size=img_size,
    #     scales=model_config['scales'],
    #     feature_map_sizes=model_config['feature_map_sizes'],
    #     aspect_ratios=model_config['aspect_ratios']
    # )
    # d_loader.load_data_from_directory(
    #     path=path, augment=False, rescale=False, rand_flip=False, rotate=False
    # )
    # m_loader = ModelLoader(
    #     aarch=model_name,
    #     num_classes=data_config['num_classes'],
    #     model_config=model_config
    # )
    # m_loader.load_model()
    # trainer = ModelTrainer(
    #     data_loader=d_loader,
    #     model_loader=m_loader,
    #     model_config=model_config,
    #     learning_rate=model_params['learning_rate'],
    #     loss=SSDLoss(model_config['neg_ratio'], data_config['num_classes'])
    # )
    # trainer.compile_model()
    # trainer.train(epochs=200)

    # best_checkpoint_path = trainer.get_best_model()
    # evaluator = ModelEvaluator(
    #     m_loader,
    #     model_config,
    #     d_loader,
    #     ckpt_path=best_checkpoint_path,
    #     phase='validation'
    # )
    # evaluator.eval()


def test_case10(path):
    config = cfg.ObjDetConfig()
    model_name = 'MobileNetV2'
    data_config = config.get_data_cfg()
    model_config = config.get_model_cfg(model_name)
    model_params = config.get_model_params(model_name)
    print('Model config')
    print(model_config)
    print('Model params')
    print(model_params)

    # label_encoder = LabelEncoder()
    img_size = model_config['input_shape'][:2]
    print('image size', img_size)
    d_loader = DataLoader(
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
    m_loader = ModelLoader(
        aarch=model_name,
        num_classes=data_config['num_classes'],
        model_config=model_config
    )
    m_loader.load_model()
    best_path = './checkpoints/MobileNetV2_ssd_last.h5'
    evaluator = ModelEvaluator(
        m_loader,
        model_config,
        d_loader,
        ckpt_path=best_path,
        phase='validation'
    )
    evaluator.eval()


def test_case11(path):
    config = cfg.ObjDetConfig()
    model_name = 'MobileNetV2'
    data_config = config.get_data_cfg()
    model_config = config.get_model_cfg(model_name)
    model_params = config.get_model_params(model_name)
    print('Model config')
    print(model_config)
    print('Model params')
    print(model_params)

    # label_encoder = LabelEncoder()
    img_size = model_config['input_shape'][:2]
    print('image size', img_size)
    d_loader = DataLoader(
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
    m_loader = ModelLoader(
        aarch=model_name,
        num_classes=data_config['num_classes'],
        model_config=model_config
    )
    m_loader.load_model()

    evaluator = ModelEvaluator(
        m_loader,
        model_config,
        d_loader,
        ckpt_path='./checkpoints/MobileNetV2_best.ckpt',
        phase='test',
        image_out_dir='D:\\output',
        draw_predictions=True,
        from_hub=False
    )
    evaluator.eval()

    # image = cv2.imread(path)
    # print(image.shape)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # success, (bboxes, classes, scores) = predictor.predict(image)
    # #class_map = {1: 'Cat', 2: 'Dog'}
    # image = vis_utils.draw_bboxes(image, bboxes, classes, scores)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #
    #cv2.imshow('output', image)
    # cv2.waitKey(0)


def test_case12(dataset_path):
    image_dir = os.path.join(dataset_path, 'images')
    test_ann_file = os.path.join(dataset_path, 'test_annotations.json')
    import json
    import cv2
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
            cv2.putText(img, cate_info['name'], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imwrite(f'test{i+1}.png', img)
        # cv2.waitKey(0)


def _perform_tests():
    path = 'C:\\Users\\Prudhvi\\Downloads\\dataset-Apple-Banana-Sandwich-Cake-Blender-Toaster-Doughnut-Pizza-Knife-Bottle'
    test_case9(path)


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
