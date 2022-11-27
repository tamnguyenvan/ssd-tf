import albumentations as A
from utils.common_defs import method_header


transforms_map = {
    'resize': A.Resize,
    'random_crop': A.RandomCrop,
    'random_fliplr': A.HorizontalFlip,
    'random_brightness': A.RandomBrightness,
    'center_crop': A.CenterCrop,
    'normalize': A.Normalize
}


@method_header(
    description='''
        build data augmentation for the images.''',
    arguments='''
        cfg : dict _description_
        ''',
    returns='''
        returns a function that can except 2 params a list and a str for bbox_params''')
def build_transforms(cfg: dict):

    transform_list = []
    if cfg is not None:
        transform_list = [transforms_map[name](
            **kwargs) for name, kwargs in cfg.items()]
    return A.Compose(transform_list, bbox_params=A.BboxParams(format='albumentations', label_fields=['classes']))
