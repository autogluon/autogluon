import math
import autogluon.core as ag
from autogluon.mxnet.optimizer import SGD
from autogluon.extra.model_zoo import EfficientNet
from autogluon.vision import ImagePredictor as task

@ag.obj(
    width_coefficient=ag.Categorical(1.1, 1.2),
    depth_coefficient=ag.Categorical(1.1, 1.2),
)
class EfficientNetB1(EfficientNet):
    def __init__(self, width_coefficient, depth_coefficient):
        input_factor = math.sqrt(2.0 / (width_coefficient ** 2) / depth_coefficient)
        input_size = math.ceil((224 * input_factor) / 32) * 32
        super().__init__(width_coefficient=width_coefficient,
                         depth_coefficient=depth_coefficient,
                         input_size=input_size)

classifier = ImagePredictor()
classifier.fit('imagenet',
               hyperparameters={
                   'net':EfficientNetB1(),
                   'search_strategy': 'grid',
                   'optimizer': SGD(learning_rate=1e-1, momentum=0.9, wd=1e-4),
                   'batch_size': 32
                   })

print(classifier.fit_summary())
