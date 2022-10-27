import sys
sys.path.append('/home/tamnv/Projects/upwork/ixolerator')
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
import meghnad.core.cv.obj_det.cfg.config as cfg
import unittest

from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.models.ssd.anchors import generate_default_boxes
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.models.ssd.utils.box_utils import compute_target


def test_case1(path):
    d_loader = DataLoader(img_size=(224, 224, 3))
    d_loader.config_connectors(path)
    d_loader.load_data_from_directory()
    m_loader = ModelLoader(aarch_name='MobileNetV2', input_shape=(224, 224, 3))
    m_loader.load_model()
    trainer = ModelTrainer(train_dataset=d_loader.train_dataset, validation_dataset=d_loader.validation_dataset,
                           test_dataset=d_loader.test_dataset, model=m_loader.model)
    trainer.compile_model()
    ret = trainer.train(epochs=100)


def test_case2(path):
    d_loader = DataLoader(img_size=(300, 300, 3))
    d_loader.load_data_from_directory(path=path)
    m_loader = ModelLoader(aarch_name='EfficientNetB1',
                           input_shape=(300, 300, 3), trainable=True)
    m_loader.load_model()
    trainer = ModelTrainer(train_dataset=d_loader.train_dataset, validation_dataset=d_loader.validation_dataset,
                           test_dataset=d_loader.test_dataset, model=m_loader.model)
    trainer.compile_model()
    ret = trainer.train(epochs=100)


def test_case3(path):
    d_loader = DataLoader(img_size=(150, 150, 3))
    d_loader.load_data_from_directory(path=path)
    m_loader = ModelLoader(aarch_name='EfficientNetB2',
                           input_shape=(150, 150, 3), trainable=True)
    m_loader.load_model()
    trainer = ModelTrainer(train_dataset=d_loader.train_dataset, validation_dataset=d_loader.validation_dataset,
                           test_dataset=d_loader.test_dataset, model=m_loader.model)
    trainer.compile_model()
    ret = trainer.train(epochs=100)


def test_case4(path):
    d_loader = DataLoader(img_size=(150, 150, 3))
    d_loader.load_data_from_directory(path=path)
    m_loader = ModelLoader(aarch_name='EfficientNetB3',
                           input_shape=(150, 150, 3), trainable=True)
    m_loader.load_model()
    trainer = ModelTrainer(train_dataset=d_loader.train_dataset, validation_dataset=d_loader.validation_dataset,
                           test_dataset=d_loader.test_dataset, model=m_loader.model)
    trainer.compile_model()
    ret = trainer.train(epochs=100)


def test_case5(path):
    d_loader = DataLoader(img_size=(150, 150, 3))
    d_loader.load_data_from_directory(path=path)
    m_loader = ModelLoader(aarch_name='EfficientNetB4',
                           input_shape=(150, 150, 3), trainable=True)
    m_loader.load_model()
    trainer = ModelTrainer(train_dataset=d_loader.train_dataset, validation_dataset=d_loader.validation_dataset,
                           test_dataset=d_loader.test_dataset, model=m_loader.model)
    trainer.compile_model()
    ret = trainer.train(epochs=100)


def test_case6(path):
    d_loader = DataLoader(img_size=(150, 150, 3))
    d_loader.load_data_from_directory(path=path)
    m_loader = ModelLoader(aarch_name='EfficientNetB5',
                           input_shape=(150, 150, 3), trainable=True)
    m_loader.load_model()
    trainer = ModelTrainer(train_dataset=d_loader.train_dataset, validation_dataset=d_loader.validation_dataset,
                           test_dataset=d_loader.test_dataset, model=m_loader.model)
    trainer.compile_model()
    ret = trainer.train(epochs=100)


def test_case7(path):
    d_loader = DataLoader(img_size=(150, 150, 3))
    d_loader.load_data_from_directory(path=path)
    m_loader = ModelLoader(aarch_name='EfficientNetB6',
                           input_shape=(150, 150, 3), trainable=True)
    m_loader.load_model()
    trainer = ModelTrainer(train_dataset=d_loader.train_dataset, validation_dataset=d_loader.validation_dataset,
                           test_dataset=d_loader.test_dataset, model=m_loader.model)
    trainer.compile_model()
    ret = trainer.train(epochs=100)


def test_case8(path):
    d_loader = DataLoader(img_size=(150, 150, 3))
    d_loader.load_data_from_directory(
        path=path, augment=True, rescale=True, rand_flip=False, rotate=False)
    m_loader = ModelLoader(aarch='EfficientNetV2S',
                           input_shape=(150, 150, 3), trainable=True)
    m_loader.load_model()
    '''
    trainer = ModelTrainer(train_dataset=d_loader.train_dataset, validation_dataset=d_loader.validation_dataset,
                           test_dataset=d_loader.test_dataset, model=m_loader.model)
    trainer.compile_model()
    trainer.train(epochs=3)
    m_loader.save_model_to_directory(trainer.model,"D://model_dir/",overwrite=True)
    infer=ModelInference(trainer.model,output_dir='D://output_data/')
    ret, predictions=infer.predict(d_loader.test_dataset,trainer.history)
    print(ret)
    if ret==IXO_RET_SUCCESS:
        infer.write_prediction(path,predictions)
    '''


def test_case9(path):
    config = cfg.ObjDetConfig()
    model_name = 'EfficientNetV2S'
    data_config = config.get_data_cfg()
    model_config = config.get_model_cfg(model_name)
    model_params = config.get_model_params(model_name)
    print('Model name:', model_name)
    print('Model config')
    print(model_config)
    print('Model params')
    print(model_params)

    # label_encoder = LabelEncoder()
    img_size = model_config['input_shape'][:2]
    print('image size', img_size)
    d_loader = DataLoader(
        batch_size=model_params['batch_size'],
        img_size=img_size,
        scales=model_config['scales'],
        feature_map_sizes=model_config['feature_map_sizes'],
        aspect_ratios=model_config['aspect_ratios']
    )
    d_loader.load_data_from_directory(
        path=path, augment=False, rescale=False, rand_flip=False, rotate=False
    )
    # for images, bboxes, labels in d_loader.train_dataset.take(1):
    #     break
    # from matplotlib import pyplot as plt
    # import cv2
    # img = images[0] * 255
    # img = img.numpy().astype(np.uint8)
    # img_h, img_w = img.shape[:2]
    # bbox = bboxes[0].numpy()
    # label = labels[0].numpy()
    # print(labels.shape)
    # print(label)
    # for b in bbox:
    #     b *= np.array([img_w, img_h, img_w, img_h])
    #     x1, y1, x2, y2 = list(map(int, b))
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow('img', img[:, :, ::-1])
    # cv2.waitKey(0)
    # image = images[0]
    # bbox = bboxes[0]
    # label = labels[0]
    # import pdb
    # pdb.set_trace()
    # default_boxes = generate_default_boxes(
    #     model_config['scales'], model_config['feature_map_sizes'], model_config['aspect_ratios'])
    # compute_target(default_boxes, bbox, label)

    m_loader = ModelLoader(
        aarch=model_name,
        num_classes=data_config['num_classes'],
        model_config=model_config
    )
    m_loader.load_model()
    trainer = ModelTrainer(
        data_loader=d_loader,
        model_loader=m_loader,
        model_config=model_config,
        learning_rate=model_params['learning_rate'],
        loss=SSDLoss(model_config['neg_ratio'], data_config['num_classes'])
    )
    trainer.compile_model()
    trainer.train(epochs=20)

    best_checkpoint_path = trainer.get_best_model()
    evaluator = ModelEvaluator(
        m_loader,
        model_config,
        d_loader,
        weights=best_checkpoint_path,
        phase='validation'
    )
    evaluator.eval()


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
        weights=best_path,
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

    m_loader = ModelLoader(
        aarch=model_name,
        num_classes=data_config['num_classes'],
        model_config=model_config,
        weights='./checkpoints/MobileNetV2_ssd_last.h5',
    )
    m_loader.load_model()

    predictor = ModelInference(
        model_loader=m_loader,
        model_config=model_config
    )
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    success, (bboxes, classes, scores) = predictor.predict(image)
    class_map = {}
    image = vis_utils.draw_bboxes(image, bboxes, classes, scores)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('output', image)
    cv2.waitKey(0)


def _perform_tests():
    path = '/home/tamnv/Downloads/dataset-Dog-Cat'
    # path = '/home/tamnv/Downloads/dataset-Dog-Cat/images/0a50fec4ab1a8354.jpg'
    test_case9(path)


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
