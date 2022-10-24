import sys
sys.path.append('/home/tamnv/Projects/upwork/ixolerator')
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.data_loader.data_loader import *
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.model_loader import *
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.train.train import *
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.train.eval import *
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.inference.pred import *
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.models.retinanet.utils.box_utils import LabelEncoder
from meghnad.core.cv.obj_det.src.backend.tensorflow_local.model_loader.models.retinanet.utils.loss_utils import RetinaNetLoss
import unittest


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
    label_encoder = LabelEncoder()
    d_loader = DataLoader(batch_size=4, img_size=(
        300, 300, 3), label_encoder=label_encoder)
    d_loader.load_data_from_directory(
        path=path, augment=False, rescale=True, rand_flip=False, rotate=False
    )
    for images, labels in d_loader.train_dataset.take(1):
        break
    print(images.shape, labels.shape)
    num_classes = 2
    m_loader = ModelLoader(
        aarch='ResNet50',
        num_classes=num_classes,
        input_shape=(300, 300, 3),
        trainable=True
    )
    m_loader.load_model()
    trainer = ModelTrainer(
        train_dataset=d_loader.train_dataset,
        validation_dataset=d_loader.validation_dataset,
        test_dataset=d_loader.test_dataset,
        model=m_loader.model,
        loss=RetinaNetLoss(num_classes),
    )
    trainer.compile_model()
    ret = trainer.train(epochs=1)


def _perform_tests():
    path = '/home/tamnv/Downloads/dataset-Dog-Cat'
    test_case9(path)


if __name__ == '__main__':
    _perform_tests()

    unittest.main()
