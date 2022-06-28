import pandas as pd
from autogluon.shift import Classifier2ST, XShiftDetector, XShiftInferrer
from autogluon.vision import ImageDataset, ImagePredictor
import os
import numpy as np
import math

def test_xsd_init():
    pred = ImagePredictor()
    xsd = XShiftDetector(pred)
    pass