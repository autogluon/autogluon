# All methods retrofitted from https://github.com/dirichletcal/dirichlet_python
# More information on methods:
# https://arxiv.org/abs/1910.12656 (Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration)
# https://arxiv.org/abs/1706.04599 (On Calibration of Modern Neural Networks)
# TODO: These methods can harm calibration of out-of-distribution data. Meant to improve calibration on in-distribution data
from .fulldirichlet import FullDirichletCalibrator
from .matrixscaling import MatrixScaling
from .tempscaling import TemperatureScaling
from .vectorscaling import VectorScaling
