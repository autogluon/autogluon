import pandas as pd
import numpy as np
import argparse
import os
import random
from autogluon.tabular import TabularPredictor
from autogluon.text.automm import AutoMMPredictor
import torch as th

def get_parser():
    parser = argparse.ArgumentParser(
        description='The Basic Example of AutoGluon for House Price Prediction.')
    parser.add_argument('--mode',
                        choices=['stack5',
                                 'weighted',
                                 'single',
                                 'automm_bag5'],
                        default='weighted',
                        help='"stack5" means 5-fold stacking. "weighted" means weighted ensemble.'
                             ' "single" means use a single model.'
                             ' "automm_bag5" means 5-fold bagging via the AutoMM model.')
    parser.add_argument('--automm-mode', choices=['ft-transformer', 'mlp'],
                        default='ft-transformer', help='Fusion model in AutoMM.')
    parser.add_argument('--text-backbone', default='google/electra-small-discriminator')
    parser.add_argument('--data_path', type=str, default='california-house-prices')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--exp_path', default=None)
    parser.add_argument('--with_tax_values', default=1, type=int)
    return parser


def get_automm_hyperparameters(mode, text_backbone):
    if mode == "ft-transformer":
        hparams = {"model.names": ["categorical_transformer",
                                   "numerical_transformer",
                                   "hf_text",
                                   "fusion_transformer"],
                   "hf_text.checkpoint_name": text_backbone}
    elif mode == "mlp":
        hparams = {"model.names": ["categorical_mlp",
                                   "numerical_mlp",
                                   "hf_text",
                                   "fusion_mlp"],
                   "hf_text.checkpoint_name": text_backbone}
    else:
        raise NotImplementedError(f"mode={mode} is not supported!")
    return hparams


def preprocess(df, with_tax_values=True, log_scale_lot=True,
               log_scale_listed_price=True, has_label=True):
    new_df = df.copy()
    new_df.drop('Id', axis=1, inplace=True)
    new_df['Elementary School'] = new_df['Elementary School'].apply(lambda ele: str(ele)[:-len(' Elementary School')] if str(ele).endswith('Elementary School') else ele)
    if log_scale_lot:
        new_df['Lot'] = np.log(new_df['Lot'] + 1)
    if log_scale_listed_price:
        log_listed_price = np.log(new_df['Listed Price']).clip(0, None)
        new_df['Listed Price'] = log_listed_price
    if with_tax_values:
        new_df['Tax assessed value'] = np.log(new_df['Tax assessed value'] + 1)
        new_df['Annual tax amount'] = np.log(new_df['Annual tax amount'] + 1)
    else:
        new_df.drop('Tax assessed value', axis=1, inplace=True)
        new_df.drop('Annual tax amount', axis=1, inplace=True)
    if has_label:
        new_df['Sold Price'] = np.log(new_df['Sold Price'])
    return new_df


def set_seed(seed):
    import torch as th
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(args):
    set_seed(args.seed)
    train_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(args.data_path, 'test.csv'))
    # For the purpose of generating submission file
    submission_df = pd.read_csv(os.path.join(args.data_path, 'sample_submission.csv'))
    train_df = preprocess(train_df,
                          with_tax_values=args.with_tax_values, has_label=True)
    test_df = preprocess(test_df,
                         with_tax_values=args.with_tax_values, has_label=False)
    label_column = 'Sold Price'
    feature_columns = [ele for ele in train_df.columns if ele != label_column]
    eval_metric = 'r2'

    automm_hyperparameters = get_automm_hyperparameters(args.automm_mode, args.text_backbone)

    tabular_hyperparameters = {
        'GBM': [
            {},
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
        ],
        'CAT': {},
        'AG_AUTOMM_NN': automm_hyperparameters,
    }

    if args.mode == 'weighted':
        predictor = TabularPredictor(eval_metric=eval_metric, label=label_column, path=args.exp_path)
        predictor.fit(train_df,
                      hyperparameters=tabular_hyperparameters)
        leaderboard = predictor.leaderboard()
        leaderboard.to_csv(os.path.join(args.exp_path, 'leaderboard.csv'))
    elif args.mode == 'single':
        predictor = AutoMMPredictor(eval_metric=eval_metric, label=label_column, path=args.exp_path)
        predictor.fit(train_df, hyperparameters=automm_hyperparameters, seed=args.seed)
    elif args.ensemble_mode == 'stack5':
        predictor = TabularPredictor(eval_metric=eval_metric, label=label_column,
                                     path=args.exp_path)
        predictor.fit(train_df,
                      hyperparameters=tabular_hyperparameters,
                      num_bag_folds=5, num_stack_levels=1)
        leaderboard = predictor.leaderboard()
        leaderboard.to_csv(os.path.join(args.exp_path, 'leaderboard.csv'))
    elif args.mode == 'automm_bag5':
        predictor = TabularPredictor(eval_metric=eval_metric, label=label_column,
                                     path=args.exp_path)
        predictor.fit(train_df,
                      hyperparameters={'AG_AUTOMM_NN': automm_hyperparameters},
                      num_bag_folds=5, num_stack_levels=0)
    else:
        raise NotImplementedError
    predictions = np.exp(predictor.predict(test_df))
    submission_df['Sold Price'] = predictions
    submission_df.to_csv(os.path.join(args.exp_path, 'submission.csv'), index=None)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.exp_path is None:
        args.exp_path = f'automm_kaggle_house_{args.ensemble_mode}_{args.mode}_{args.text_backbone}'
    th.manual_seed(args.seed)
    train(args)
