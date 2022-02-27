import os
import json
import argparse
import numpy as np
import random
import pandas as pd
import copy
from autogluon.text import TextPredictor
from autogluon.tabular import TabularPredictor
from autogluon.core.utils.loaders import load_pd
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config


TASKS = \
    {'cola': (['sentence'], 'label', 'mcc', ['mcc']),
     'sst': (['sentence'], 'label', 'acc', ['acc']),
     'mrpc': (['sentence1', 'sentence2'], 'label', 'acc', ['acc', 'f1']),
     'sts': (['sentence1', 'sentence2'], 'score', 'rmse', ['pearsonr', 'spearmanr']),
     'qqp': (['sentence1', 'sentence2'], 'label', 'acc', ['acc', 'f1']),
     'mnli_m': (['sentence1', 'sentence2'], 'label', 'acc', ['acc']),
     'mnli_mm': (['sentence1', 'sentence2'], 'label', 'acc', ['acc']),
     'qnli': (['question', 'sentence'], 'label', 'acc', ['acc']),
     'rte': (['sentence1', 'sentence2'], 'label', 'acc', ['acc']),
     'wnli': (['sentence1', 'sentence2'], 'label', 'acc', ['acc']),
     'snli': (['sentence1', 'sentence2'], 'label', 'acc', ['acc'])}


def get_parser():
    parser = argparse.ArgumentParser(description='The Basic Example of AutoML for Text Prediction.')
    parser.add_argument('--train_file', type=str,
                        help='The training pandas dataframe.',
                        default=None)
    parser.add_argument('--dev_file', type=str,
                        help='The validation pandas dataframe',
                        default=None)
    parser.add_argument('--test_file', type=str,
                        help='The test pandas dataframe',
                        default=None)
    parser.add_argument('--seed', type=int,
                        help='The seed',
                        default=123)
    parser.add_argument('--feature_columns', help='Feature columns', default=None)
    parser.add_argument('--label_columns', help='Label columns', default=None)
    parser.add_argument('--eval_metric', type=str,
                        help='The metric used to evaluate the model.',
                        default=None)
    parser.add_argument('--all_metrics', type=str,
                        help='All metrics that we will report. This will usually '
                             'include the eval_metric',
                        default=None)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to train the model')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to evaluate the model')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='The experiment directory where the model params will be written.')
    parser.add_argument('--mode',
                        choices=['stacking', 'weighted', 'single'],
                        default='single',
                        help='Whether to use a single model or a stack ensemble. '
                             'If it is "single", If it is turned on, we will use 5-fold, 1-layer for stacking.')
    parser.add_argument('--preset', type=str,
                        help='Pre-registered configurations',
                        choices=['medium_quality_faster_train',
                                 'high_quality',
                                 'best_quality'],
                        default=None)
    return parser


def train(args):
    if args.task is not None:
        feature_columns, label_column, eval_metric, all_metrics = TASKS[args.task]
    else:
        raise NotImplementedError
    if args.exp_dir is None:
        args.exp_dir = 'autogluon_text_{}'.format(args.task)
    train_df = load_pd.load(args.train_file)
    dev_df = load_pd.load(args.dev_file)
    test_df = load_pd.load(args.test_file)
    train_df = train_df[feature_columns + [label_column]]
    dev_df = dev_df[feature_columns + [label_column]]
    test_df = test_df[feature_columns]
    if args.task == 'mrpc' or args.task == 'sts':
        # Augmenting the un-ordered set manually.
        train_df_other_part = pd.DataFrame({feature_columns[0]: train_df[feature_columns[1]],
                                            feature_columns[1]: train_df[feature_columns[0]],
                                            label_column: train_df[label_column]})
        real_train_df = pd.concat([train_df, train_df_other_part])
        real_dev_df = dev_df
    else:
        real_train_df = train_df
        real_dev_df = dev_df

    hyperparameters = get_hyperparameter_config('multimodal')
    if args.preset is not None and args.mode in ['stacking', 'weighted']:
        hyperparameters['AG_TEXT_NN']['presets'] = args.preset

    if args.mode == 'stacking':
        predictor = TabularPredictor(label=label_column,
                                     eval_metric=eval_metric,
                                     path=args.exp_dir)
        predictor.fit(train_data=real_train_df,
                      tuning_data=real_dev_df,
                      hyperparameters=hyperparameters,
                      num_bag_folds=5,
                      num_stack_levels=1)
    elif args.mode == 'weighted':
        predictor = TabularPredictor(label=label_column,
                                     eval_metric=eval_metric,
                                     path=args.exp_dir)
        predictor.fit(train_data=real_train_df,
                      tuning_data=real_dev_df,
                      hyperparameters=hyperparameters)
    elif args.mode == 'single':
        # When no embedding is used,
        # we will just use TextPredictor that will train a single model internally.
        predictor = TextPredictor(label=label_column,
                                  eval_metric=eval_metric,
                                  path=args.exp_dir)
        predictor.fit(train_data=real_train_df,
                      tuning_data=real_dev_df,
                      presets=args.preset,
                      seed=args.seed)
    else:
        raise NotImplementedError
    dev_metric_score = predictor.evaluate(dev_df)
    dev_predictions = predictor.predict(dev_df, as_pandas=True)
    test_predictions = predictor.predict(test_df, as_pandas=True)
    dev_predictions.to_csv(os.path.join(args.exp_dir, 'dev_prediction.csv'))
    test_predictions.to_csv(os.path.join(args.exp_dir, 'test_prediction.csv'))
    with open(os.path.join(args.exp_dir, 'final_model_scores.json'), 'w') as of:
        json.dump({f'valid_{eval_metric}': dev_metric_score}, of)


def predict(args):
    if args.use_tabular:
        predictor = TabularPredictor.load(args.model_dir)
    else:
        predictor = TextPredictor.load(args.model_dir)
    test_prediction = predictor.predict(args.test_file, as_pandas=True)
    if args.exp_dir is None:
        args.exp_dir = '.'
    test_prediction.to_csv(os.path.join(args.exp_dir, 'test_prediction.csv'))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.do_train:
        train(args)
    if args.do_eval:
        predict(args)
