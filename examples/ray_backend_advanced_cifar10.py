import logging

import autogluon as ag
from autogluon import image_classification as task

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    dataset = task.Dataset('./CIFAR10/train', './CIFAR10/valid')

    models, best_result, search_space = task.fit(dataset,
                                                 nets=ag.Nets(
                                                     [task.model_zoo.get_model('resnet18_v1'),
                                                      task.model_zoo.get_model('resnet34_v1'),
                                                      task.model_zoo.get_model('resnet50_v1'),
                                                      task.model_zoo.get_model('resnet101_v1'),
                                                      task.model_zoo.get_model('resnet152_v1')]),
                                                 optimizers=ag.Optimizers(
                                                     [ag.optims.get_optim('sgd'),
                                                      ag.optims.get_optim('adam')]),
                                                 backend='ray')
    logger.debug('trials results:')
    logger.debug(models)
    logger.debug('=========================')
    logger.debug('best results:')
    logger.debug(best_result)
    logger.debug('=========================')
    logger.debug('print search space')
    logger.debug(search_space)
