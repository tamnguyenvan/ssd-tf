import meghnad.core.cv.obj_det.src.backend.tensorflow_local as tfl
import meghnad.core.cv.obj_det.src.backend.pytorch as pyl
from utils.ret_values import *

class image_detector:
    def __init__(self, arch):
        self.module = tfl
        if arch == 'pytorch':
            self.module = pyl
        self.data_loader = None
        self.model_loader = None
        self.model_trainer = None

    def DataLoader(self, config_obj):
        user_config = config_obj.get_data_cfg()
        batch_size = 1
        if user_config['batch_size']:
            batch_size=user_config['batch_size']
        img_size = (128, 128, 3)
        if user_config['img_size']:
            img_size=user_config['img_size']
        train_dir=None
        if user_config['train_dir']:
            train_dir = user_config['train_dir']
        test_dir = None
        if user_config['test_dir']:
            test_dir = user_config['test_dir']
        val_dir = None
        if user_config['val_dir']:
            val_dir = user_config['val_dir']
        train_test_val_split = (0.7, 0.2, 0.1)
        if user_config['train_test_val_split']:
            train_test_val_split = user_config['train_test_val_split']
        self.data_loader = self.module.DataLoader(batch_size, img_size, train_dir, test_dir,
                                                  val_dir, train_test_val_split)
        return IXO_RET_SUCCESS

    def ModelLoader(self, config_obj):
        user_config=config_obj.get_model_cfg()
        aarch_name='MobileNet'
        if user_config['model']:
            aarch_name=user_config['model']
        saved_weights_path = None
        if user_config['saved_weights_path']:
            saved_weights_path=user_config['saved_weights_path']
        initialize_weight=False
        if user_config['initialize_weight']:
            saved_weights_path=user_config['initialize_weight']
        input_shape = (320, 320, 3)
        if user_config['input_shape']:
            saved_weights_path=user_config['input_shape']
        trainable = False
        if user_config['trainable']:
            saved_weights_path=user_config['trainable']
        pooling_type = None
        if user_config['trainable']:
            saved_weights_path=user_config['trainable']
        num_classes = 4
        if user_config['num_classes']:
            saved_weights_path=user_config['num_classes']
        saved_model_path = None
        if user_config['saved_model_path']:
            saved_weights_path=user_config['saved_model_path']
        self.model_loader = self.module.ModelLoader(aarch_name,saved_weights_path,
                                                    initialize_weight, input_shape, trainable,
                                                    pooling_type, num_classes, saved_model_path)
        return IXO_RET_SUCCESS

    def ModelTrainer(self, config_obj):
        model_params= config_obj.get_model_params()
        user_config= config_obj.get_user_config()
        loss='mae'
        if model_params['loss']:
            loss=model_params['loss']
        metrics=["accuracy"]
        if model_params['metrics']:
            metrics=model_params['metrics']
        learning_rate = 0.0001
        if model_params['learning_rate']:
            metrics=model_params['learning_rate']
        optimizer="Adam"
        if model_params['optimizer']:
            metrics = model_params['optimizer']
        store_tensorboard_logs = True
        if user_config['store_tensorboard_logs']:
            metrics = user_config['store_tensorboard_logs']
        log_dir = None
        if user_config['log_dir']:
            metrics = user_config['log_dir']
        prediction_postprocessing = None
        if user_config['prediction_postprocessing']:
            metrics = user_config['prediction_postprocessing']


        self.model_trainer = self.module.Train.ModelTrainer(self.data_loader.train_dataset,
                                                            self.data_loader.validation_dataset,
                                                            self.data_loader.test_dataset,
                                                            self.model_loader.model,
                                                            loss,
                                                            metrics, learning_rate, optimizer, store_tensorboard_logs,
                                                            log_dir, prediction_postprocessing)
        return IXO_RET_SUCCESS

    def load_data_from_directory(self, path):
        return self.data_loader.load_data_from_directory(path)

    def load_data_from_url(self, url, save_as=None, data_dir=None):
        return self.data_loader.load_data_from_url(url, save_as, data_dir)

    def augment_data(self, rescale=True, random_flip=False, random_rotation=False):
        return self.data_loader.augment_data(rescale, random_flip, random_rotation)

    def read_data(self, path, annotation_file):
        return self.data_loader.read_data(path, annotation_file)

    def load_model(self):
        return self.model_loader.load_model()

    def load_model_from_url(self, url, model_dir=None):
        return self.model_loader.load_model_from_url(url, model_dir)

    def load_model_from_directory(self, filepath, compile):
        return self.model_loader.load_model_from_directory(filepath, compile)

    def save_model_to_directory(self, file_path, overwrite):
        return self.model_loader.save_model_to_directory(file_path, overwrite)

    def compile_model(self):
        return self.model_trainer.compile_model()

    def train(self, epochs):
        return self.model_trainer.train(epochs)

    def predict(self, model, prediction_postprocessing, input_image_batch, history):
        return self.module.ModelInference(model, prediction_postprocessing).predict(self, input_image_batch, history)

    def evaluate(self, model, test_dataset):
        return self.module.ModelEvaluator(model, test_dataset).eval()
