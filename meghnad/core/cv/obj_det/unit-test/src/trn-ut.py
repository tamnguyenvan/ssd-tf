import sys
sys.path.append('D:\\10-27-22\ixolerator')
import os


import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus, 'GPU')

import cv2

from meghnad.core.cv.obj_det.src.tensorflow.train import TFObjDetTrn
from meghnad.core.cv.obj_det.src.tensorflow.data_loader import TFObjDetDataLoader
from meghnad.core.cv.obj_det.cfg import ObjDetConfig
import unittest


def test_case1():
    settings = ['light']
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    trainer = TFObjDetTrn(settings=settings)
    trainer.config_connectors(path)
    trainer.train(epochs=10)


def test_case2():
    path = 'C:\\Users\\Prudhvi\\Downloads\\grocery_dataset'
    cfg_obj = ObjDetConfig()
    data_cfg = cfg_obj.get_data_cfg()
    model_cfg = cfg_obj.get_model_cfg('MobileNetV2')
    dataloader = TFObjDetDataLoader(path)

    for images, gt_confs, gt_locs in dataloader.train_dataset.take(1):
        break
    print(images.shape)


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
            cv2.putText(img, cate_info['name'], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imwrite(f'test{i+1}.png', img)


def _perform_tests():
    test_case1()


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
