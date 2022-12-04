from typing import Dict
import importlib

from tensorflow.keras import Model


def build_model(cfg: Dict):
    assert 'arch' in cfg, 'Model arch is unknown'

    model_arch = cfg['arch']
    if 'backbone' in cfg and cfg['backbone']:
        backbone = cfg['backbone']
        module_name = f'{model_arch}.{model_arch}_{backbone}'
        model_fn_name = f'{model_arch}_{backbone}'
    else:
        module_name = f'{model_arch}.{model_arch}'
        model_fn_name = f'{model_arch}_{backbone}'

    name = f'{__name__}.{module_name}'
    module = importlib.import_module(name)
    model = module.__dict__[model_fn_name](cfg)
    setattr(model, 'cfg', cfg)
    return model


def build_eval_model(base_model: Model) -> Model:
    cfg = base_model.cfg
    assert 'arch' in cfg, 'Model arch is unknown'

    model_arch = cfg['arch']
    module_name = f'{model_arch}.decoder'

    name = f'{__name__}.{module_name}'
    module = importlib.import_module(name)
    decoder_fn_name = module.__all__[0]
    outputs = module.__dict__[decoder_fn_name](cfg)(base_model.output)
    # bboxes, classes, scores = SSDDecoder()(base_model.output)
    return Model(inputs=base_model.input, outputs=outputs)


def build_losses(cfg: Dict):
    assert 'arch' in cfg, 'Model arch is unknown'
    assert 'loss' in cfg, 'Loss function is unknown'
    model_arch = cfg['arch']

    module_name = f'{__name__}.{model_arch}.losses'
    module = importlib.import_module(module_name)
    loss_cls = module.__dict__[cfg['loss']]
    loss_obj = loss_cls(cfg)
    loss_funcs = [getattr(loss_obj, func) for func in dir(
        loss_cls) if callable(getattr(loss_cls, func)) and not func.startswith('__')]
    return loss_funcs
