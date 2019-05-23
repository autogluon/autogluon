import logging

import autogluon as ag
from autogluon import image_classification as task

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    dataset = task.Dataset(name='CIFAR10')

    trials, best_result, search_space = task.fit(dataset,
                                                 nets=ag.Nets(
                                                     [task.model_zoo.get_model('resnet18_v1'),
                                                      task.model_zoo.get_model('resnet34_v1'),
                                                      task.model_zoo.get_model('resnet50_v1'),
                                                      task.model_zoo.get_model('resnet101_v1'),
                                                      task.model_zoo.get_model('resnet152_v1')]),
                                                 optimizers=ag.Optimizers(
                                                     [ag.optim.get_optim('sgd'),
                                                      ag.optim.get_optim('adam')]))
    logger.debug('Best Results:')
    logger.debug(best_result)
    logger.debug('=========================')
    logger.debug('Print search space:')
    logger.debug(search_space)
