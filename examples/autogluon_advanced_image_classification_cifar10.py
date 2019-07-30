import logging

import autogluon as ag
from autogluon import image_classification as task

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    dataset = task.Dataset(name='CIFAR10')

    results = task.fit(dataset,
                       nets=ag.Nets(['cifar_resnet20_v1',
                                     'cifar_resnet56_v1',
                                     'cifar_resnet110_v1']),
                       optimizers=ag.Optimizers(['sgd', 'adam']))

    logger.debug('Best result:')
    logger.debug(results.metric)
    logger.debug('=========================')
    logger.debug('Best search space:')
    logger.debug(results.config)
    logger.debug('=========================')
    logger.debug('Total time cost:')
    logger.debug(results.time)
