import logging

from autogluon import named_entity_recognition as task


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    dataset = task.Dataset(name='CoNLL2003',
                           train_path='/home/ubuntu/conll2003/train.txt',
                           val_path='/home/ubuntu/conll2003/test.txt',
                           indexes_format={0: 'text', 3: 'ner'},
                           max_sequence_length=180,
                           batch_size=8)

    nets = task.utils.prepare_nets(['bert_12_768_12', 'bert_24_1024_16'], dropout_list=[(0.0, 0.5), (0.0, 0.5)])

    optims = task.utils.prepare_optims(['adam', 'bertadam'], lr_list=[(10 ** -5, 10 ** -4), (10 ** -5, 10 ** -4)])

    stop_criterion = {
        'time_limits': 15000,
        'max_metric': 0.95,
        'max_trial_count': 3
    }

    resources_per_trial = {
        'max_num_gpus': 4,
        'max_num_cpus': 4,
        'max_training_epochs': 5
    }

    results = task.fit(dataset,
                       nets=nets,
                       optimizers=optims,
                       resources_per_trial=resources_per_trial,
                       stop_criterion=stop_criterion)

    logger.info('Best result:')
    logger.info(results.metric)
    logger.info('=========================')
    logger.info('Best search space:')
    logger.info(results.config)
