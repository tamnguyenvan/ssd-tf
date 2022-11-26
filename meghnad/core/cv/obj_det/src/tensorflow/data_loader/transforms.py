import albumentations as A


transforms_map = {
    'resize': A.Resize,
    'random_crop': A.RandomCrop,
    'random_fliplr': A.HorizontalFlip,
    'random_brightness': A.RandomBrightness,
    'center_crop': A.CenterCrop,
    'normalize': A.Normalize
}


def build_transforms(cfg: dict):
    """_summary_

    Parameters
    ----------
    cfg : dict
        _description_
    """
    transform_list = []
    if cfg is not None:
        transform_list = [transforms_map[name](
            **kwargs) for name, kwargs in cfg.items()]
    return A.Compose(transform_list, bbox_params=A.BboxParams(format='albumentations', label_fields=['classes']))
