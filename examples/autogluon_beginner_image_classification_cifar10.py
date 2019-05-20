import logging

from autogluon import image_classification as task


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    dataset = task.Dataset('./CIFAR10/train', './CIFAR10/valid')
    models = task.fit(dataset)

    logger.debug('Best result:')
    logger.debug(models[1])
    logger.debug('=========================')
    logger.debug('Best search space:')
    logger.debug(models[2])
