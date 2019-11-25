import logging
import autogluon as ag
from autogluon import TextClassification as task

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def test_text_classification():
    logger.debug('Start testing autogluon text classification')
    dataset = task.Dataset(name='SST')
    logger.debug(dataset)
    logger.debug('Finished.')
    ag.done()


if __name__ == '__main__':
    test_text_classification()
