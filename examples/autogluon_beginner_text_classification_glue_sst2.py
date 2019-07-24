import logging

from autogluon import Linear
from autogluon import Nets, Optimizers, get_optim
from autogluon import text_classification as task
from autogluon.task.text_classification import get_model

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    nets_to_search = Nets([
        get_model('bert_12_768_12', **{'classification_layers': Linear('dense', lower=0, upper=3)}),
        get_model('bert_24_1024_16', **{'classification_layers': Linear('dense', lower=0, upper=3)})
    ])

    optims = default_optimizers = Optimizers([
        get_optim('adam')
        # get_optim('sgd'),
        # get_optim('ftml'),
        # get_optim('bertadam')
    ])

    default_stop_criterion = {
        'time_limits': 5 * 60 * 60,
        'max_metric': 0.80,  # TODO Should be place a bound on metric?
        'max_trial_count': 10
    }

    default_resources_per_trial = {
        'max_num_gpus': 2,
        'max_num_cpus': 16,
        'max_training_epochs': 10
    }

    dataset = task.Dataset(name='glue_sst', data_format='tsv', field_indices=[0, 1])
    results = task.fit(dataset,
                       nets=nets_to_search,
                       optimizers=optims,
                       stop_criterion=default_stop_criterion,
                       resources_per_trial=default_resources_per_trial,
                       savedir='/home/ubuntu/AutoGluon/checkpoint/glue_sst2')

    logger.info('Best result:')
    logger.info(results.metric)
    logger.info('=========================')
    logger.info('Best search space:')
    logger.info(results.config)
