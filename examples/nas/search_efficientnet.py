import math
import autogluon as ag
from autogluon import ImageClassification as task

@ag.obj(
    width_coefficient=ag.space.Categorical(1.1, 1.2),
    depth_coefficient=ag.space.Categorical(1.1, 1.2),
)
class EfficientNetB1(ag.nas.EfficientNet):
    def __init__(self, width_coefficient, depth_coefficient):
        input_factor = math.sqrt(2.0 / (width_coefficient ** 2) / depth_coefficient)
        input_size = math.ceil((224 * input_factor) / 32) * 32
        super().__init__(width_coefficient=width_coefficient,
                         depth_coefficient=depth_coefficient,
                         input_size=input_size)

results = task.fit('imagenet', net=EfficientNetB1(), search_strategy='grid',
                   optimizer=ag.optimizer.SGD(learning_rate=1e-1,momentum=0.9,wd=1e-4),
                   batch_size=32)

print(results)
