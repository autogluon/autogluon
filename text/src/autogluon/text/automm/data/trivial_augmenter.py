"""
This file implements TrivialAugment.(https://arxiv.org/abs/2103.10158) We extend it for multi-modality setting. 

Code is partically adapted from its official implementation https://github.com/automl/trivialaugment
"""

import random
from selectors import EpollSelector
from PIL import ImageOps, ImageEnhance, ImageFilter, Image, ImageDraw
import nlpaug.augmenter.word as naw
from .utils import InsertPunctuation
import nltk


def scale_parameter(level, maxval, type):
    """
    Helper function to scale `val` between 0 and maxval .

    Parameters
    ----------
    level: Level of the operation that will be between [0, image_parameter_max].
    maxval: Maximum value that the operation can have.
    type: return float or int

    Returns
    -------
    An adjust scale
    """
    if type == "float":
        return float(level) * maxval / image_parameter_max
    elif type == "int":
        return int(level * maxval / image_parameter_max)


class TransformT(object):
    """
    Each instance of this class represents a specific transform.
    """

    def __init__(self, name, xform_fn):
        """
        Parameters
        ----------
        name: name of the operation
        xform_fn: augmentation operation function
        """
        self.name = name
        self.xform = xform_fn

    def __repr__(self):
        return "<" + self.name + ">"

    def augment(self, level, data):
        return self.xform(data, level)


identity = TransformT("identity", lambda data, level: data)


auto_contrast = TransformT("AutoContrast", lambda pil_img, level: ImageOps.autocontrast(pil_img))


equalize = TransformT("Equalize", lambda pil_img, level: ImageOps.equalize(pil_img))


def _rotate_impl(pil_img, level):
    """
    Rotates `pil_img` from -30 to 30 degrees depending on `level`.
    """
    max = 30
    degrees = scale_parameter(level, max, "int")
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees)


rotate = TransformT("Rotate", _rotate_impl)


def _solarize_impl(pil_img, level):
    """
    Applies PIL Solarize to `pil_img` with strength level.
    """
    max = 256
    level = scale_parameter(level, max, "int")
    return ImageOps.solarize(pil_img, max - level)


solarize = TransformT("Solarize", _solarize_impl)


def _posterize_impl(pil_img, level):
    """
    Applies PIL Posterize to `pil_img` with strength level.
    """
    max = 4
    min = 0
    level = scale_parameter(level, max - min, "int")
    return ImageOps.posterize(pil_img, max - level)


posterize = TransformT("Posterize", _posterize_impl)


def _enhancer_impl(enhancer):
    """
    Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL.
    """
    min = 0.1
    max = 1.8

    def impl(pil_img, level):
        v = scale_parameter(level, max - min, "float") + min  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


color = TransformT("Color", _enhancer_impl(ImageEnhance.Color))

contrast = TransformT("Contrast", _enhancer_impl(ImageEnhance.Contrast))

brightness = TransformT("Brightness", _enhancer_impl(ImageEnhance.Brightness))

sharpness = TransformT("Sharpness", _enhancer_impl(ImageEnhance.Sharpness))


def _shear_x_impl(pil_img, level):
    """
    Shears the image along the horizontal axis with `level` magnitude.
    """
    max = 0.3
    level = scale_parameter(level, max, "float")
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))


shear_x = TransformT("ShearX", _shear_x_impl)


def _shear_y_impl(pil_img, level):
    """
    Shear the image along the vertical axis with `level` magnitude.
    """
    max = 0.3
    level = scale_parameter(level, max, "float")
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))


shear_y = TransformT("ShearY", _shear_y_impl)


def _translate_x_impl(pil_img, level):
    """
    Translate the image in the horizontal direction by `level` number of pixels.
    """
    max = 10
    level = scale_parameter(level, max, "int")
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))


translate_x = TransformT("TranslateX", _translate_x_impl)


def _translate_y_impl(pil_img, level):
    """
    Translate the image in the vertical direction by `level` number of pixels.
    """
    max = 10
    level = scale_parameter(level, max, "int")
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))


translate_y = TransformT("TranslateY", _translate_y_impl)


def set_image_augmentation_space(max_strength):
    global image_parameter_max
    image_parameter_max = max_strength
    image_all_transform = [
        identity,
        auto_contrast,
        equalize,
        rotate,  # extra coin-flip
        solarize,
        color,  # enhancer
        posterize,
        contrast,  # enhancer
        brightness,  # enhancer
        sharpness,  # enhancer
        shear_x,  # extra coin-flip
        shear_y,  # extra coin-flip
        translate_x,  # extra coin-flip
        translate_y,  # extra coin-flip
    ]
    return image_all_transform


def set_text_augmentation_space(max_strength):
    assert max_strength <= 1.0, "Invalid maximum strength. Must <= 1.0"
    global text_parameter_max
    text_parameter_max = max_strength
    text_all_transform = [
        "identity",
        "syn_replacement",
        "random_delete",
        "random_swap",
        "insert_punc",
    ]

    try:
        nltk.data.find("tagger/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")

    return text_all_transform


class TrivialAugment:
    """
    Implementation for TrivialAugment (https://arxiv.org/abs/2103.10158)
    Random sample one operation from all_transform
    Random a strength between [0, max_strength]
    """

    def __init__(self, datatype, max_strength) -> None:
        """
        Parameters
        ----------
        datatype
            Modality type, currently support "text" and "img"
        max_strength
            Max strength for augmentation operation. 
        """
        assert max_strength > 0, "Invalid maximum strength. Must > 0"
        self.max_strength = max_strength
        self.data_type = datatype
        if datatype == "img":
            self.all_transform = set_image_augmentation_space(self.max_strength)
        elif datatype == "text":
            self.all_transform = set_text_augmentation_space(self.max_strength)
        else:
            raise NotImplementedError

    def __call__(self, data):
        if self.data_type == "img":
            return self.augment_image(data)
        elif self.data_type == "text":
            return self.augment(data)

    def augment_image(self, data):
        op = random.choice(self.all_transform)
        scale = random.randint(0, self.max_strength)
        return op.augment(scale, data)

    def augment(self, data):
        op_str = random.choice(self.all_transform)
        scale = random.uniform(0, self.max_strength)
        if op_str == "identity":
            return data
        elif op_str == "syn_replacement":
            op = naw.SynonymAug(aug_src="wordnet", aug_p=scale, aug_max=None)
        elif op_str == "random_swap":
            op = naw.RandomWordAug(action="swap", aug_p=scale, aug_max=None)
        elif op_str == "random_delete":
            op = naw.RandomWordAug(action="delete", aug_p=scale, aug_max=None)
        elif op_str == "insert_punc":
            op = InsertPunctuation(aug_p=scale, aug_max=None)
        else:
            raise NotImplementedError
        return op.augment(data)
