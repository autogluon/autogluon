import logging

from autogluon import text_classification as task


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    dataset = task.Dataset(name='sst_2', data_format='json')
    results = task.fit(dataset)

    logger.info('Best result:')
    logger.info(results.metric)
    logger.info('=========================')
    logger.info('Best search space:')
    logger.info(results.config)
