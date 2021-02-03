# code adapted from:
# https://github.com/kakaobrain/fast-autoaugment
# https://github.com/rpmcruz/autoaugment
import random

import numpy as np
from collections import defaultdict
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from autogluon.mxnet.utils.pil_transforms import ToPIL, ToNDArray

from mxnet.gluon.block import Block

random_mirror = True

RESAMPLE_MODE=PIL.Image.BICUBIC

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0),
                         RESAMPLE_MODE)


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0),
                         RESAMPLE_MODE)


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0),
                         RESAMPLE_MODE)


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v),
                         RESAMPLE_MODE)


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0),
                         RESAMPLE_MODE)


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v),
                         RESAMPLE_MODE)


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = PIL.Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    #assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)



def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v),
                         resample=RESAMPLE_MODE)


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0),
                         resample=RESAMPLE_MODE)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)



def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l



augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

PARAMETER_MAX = 10


def float_parameter(level, maxval):
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    return int(float_parameter(level, maxval))


def autoaug2fastaa(f):
    def autoaug():
        mapper = defaultdict(lambda: lambda x: x)
        mapper.update({
            'ShearX': lambda x: float_parameter(x, 0.3),
            'ShearY': lambda x: float_parameter(x, 0.3),
            'TranslateX': lambda x: int_parameter(x, 10),
            'TranslateY': lambda x: int_parameter(x, 10),
            'Rotate': lambda x: int_parameter(x, 30),
            'Solarize': lambda x: 256 - int_parameter(x, 256),
            'Posterize2': lambda x: 4 - int_parameter(x, 4),
            'Contrast': lambda x: float_parameter(x, 1.8) + .1,
            'Color': lambda x: float_parameter(x, 1.8) + .1,
            'Brightness': lambda x: float_parameter(x, 1.8) + .1,
            'Sharpness': lambda x: float_parameter(x, 1.8) + .1,
            'CutoutAbs': lambda x: int_parameter(x, 20)
        })

        def low_high(name, prev_value):
            _, low, high = get_augment(name)
            return float(prev_value - low) / (high - low)

        policies = f()
        new_policies = []
        for policy in policies:
            new_policies.append([(name, pr, low_high(name, mapper[name](level))) for name, pr, level in policy])
        return new_policies

    return autoaug


def rand_augment_list():  # 16 oeprations and their ranges
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l


@autoaug2fastaa
def autoaug_cifar10_policies():
    return [
        [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
        [('Rotate', 0.7, 2), ('TranslateXAbs', 0.3, 9)],
        [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
        [('ShearY', 0.5, 8), ('TranslateYAbs', 0.7, 9)],
        [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)],
        [('ShearY', 0.2, 7), ('Posterize2', 0.3, 7)],
        [('Color', 0.4, 3), ('Brightness', 0.6, 7)],
        [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
        [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
        [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)],
        [('Color', 0.7, 7), ('TranslateXAbs', 0.5, 8)],
        [('Equalize', 0.3, 7), ('AutoContrast', 0.4, 8)],
        [('TranslateYAbs', 0.4, 3), ('Sharpness', 0.2, 6)],
        [('Brightness', 0.9, 6), ('Color', 0.2, 6)],
        [('Solarize', 0.5, 2), ('Invert', 0.0, 3)],
        [('Equalize', 0.2, 0), ('AutoContrast', 0.6, 0)],
        [('Equalize', 0.2, 8), ('Equalize', 0.6, 4)],
        [('Color', 0.9, 9), ('Equalize', 0.6, 6)],
        [('AutoContrast', 0.8, 4), ('Solarize', 0.2, 8)],
        [('Brightness', 0.1, 3), ('Color', 0.7, 0)],
        [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
        [('TranslateYAbs', 0.9, 9), ('TranslateYAbs', 0.7, 9)],
        [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],
        [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
        [('TranslateYAbs', 0.7, 9), ('AutoContrast', 0.9, 1)],
    ]

@autoaug2fastaa
def autoaug_imagenet_policies():
    return [
        [('Posterize2', 0.4, 8), ('Rotate', 0.6, 9)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        [('Posterize2', 0.6, 7), ('Posterize2', 0.6, 6)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
        [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
        [('Posterize2', 0.8, 5), ('Equalize', 1.0, 2)],
        [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
        [('Equalize', 0.6, 8), ('Posterize2', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
        [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Color', 1.0, 0)],
        [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
        [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
    ]


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)

class Augmentation(object):
    def __init__(self, policies):
        """
        plicies : list of (name, pr, level)
        """
        self.policies = policies
        self.topil = ToPIL()

    def __call__(self, img):
        img = self.topil(img)
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img

class AugmentationBlock(Block):
    r"""
    AutoAugment Block

    Example
    -------
    >>> from autogluon.vision.utils.augment import AugmentationBlock, autoaug_imagenet_policies
    >>> aa_transform = AugmentationBlock(autoaug_imagenet_policies())
    """
    def __init__(self, policies):
        """
        plicies : list of (name, pr, level)
        """
        super().__init__()
        self.policies = policies
        self.topil = ToPIL()
        self.tond = ToNDArray()

    def forward(self, img):
        img = self.topil(img)
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        img = self.tond(img)
        return img

class RandAugment(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = rand_augment_list()
        self.topil = ToPIL()

    def __call__(self, img):
        img = self.topil(img)
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            if random.random() > random.uniform(0.2, 0.8):
                continue
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img

