import logging

from autogluon import image_classification as task


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    dataset = task.Dataset(name='CIFAR10')
    results = task.fit(dataset)

    logger.debug('Best result:')
    logger.debug(results.val_accuracy)
    logger.debug('=========================')
    logger.debug('Best search space:')
    logger.debug(results.config)
    logger.debug('=========================')
    logger.debug('Total time cost:')
    logger.debug(results.time)
