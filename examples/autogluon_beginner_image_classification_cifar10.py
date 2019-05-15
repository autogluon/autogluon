import logging

from autogluon import image_classification as task


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    dataset = task.Dataset('./CIFAR10/train', './CIFAR10/valid')
    models = task.fit(dataset)

    logger.debug('trials results:')
    logger.debug(models[0])
