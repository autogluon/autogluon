import os
import argparse
import json
from autogluon.text.automm import AutoMMPredictor

from dataset import (
    AdultTabularDataset,
    AloiTabularDataset,
    CaliforniaHousingTabularDataset,
    CovtypeTabularDataset,
    EpsilonTabularDataset,
    HelenaTabularDataset,
    HiggsSmallTabularDataset,
    JannisTabularDataset,
    MicrosoftTabularDataset,
    YahooTabularDataset,
    YearTabularDataset,
)

TABULAR_DATASETS = {
    'ad': AdultTabularDataset,
    'al': AloiTabularDataset,
    'ca': CaliforniaHousingTabularDataset,
    'co': CovtypeTabularDataset,
    'ep': EpsilonTabularDataset,
    'he': HelenaTabularDataset,
    'hi': HiggsSmallTabularDataset,
    'ja': JannisTabularDataset,
    'mi': MicrosoftTabularDataset,
    'ya': YahooTabularDataset,
    'ye': YearTabularDataset,
}

automm_hyperparameters = {
    'data.categorical.convert_to_text': False,
    'model.names': [
        "categorical_transformer",
        "numerical_transformer",
        "fusion_transformer"
    ],
    'model.numerical_transformer.embedding_arch': ['linear'],
    'env.batch_size': 128,
    'env.per_gpu_batch_size': 128,
    'env.num_workers': 12,
    'env.num_workers_evaluation': 12,
    'env.num_gpus': 1,
    'optimization.max_epochs': 2000,  # Specify a large value to train until convergence
    'optimization.weight_decay': 1.0e-5,
    'optimization.lr_choice': None,
    'optimization.lr_schedule': "polynomial_decay",
    'optimization.warmup_steps': 0.,
    'optimization.patience': 20,
    'optimization.top_k': 3,
}


def main(args):
    assert args.dataset_name in TABULAR_DATASETS.keys(), 'Unsupported dataset name.'
    
    ### Dataset loading
    train_data = TABULAR_DATASETS[args.dataset_name](
        'train', args.dataset_dir
    )
    
    val_data = TABULAR_DATASETS[args.dataset_name](
        'val', args.dataset_dir
    )
    
    test_data = TABULAR_DATASETS[args.dataset_name](
        'test', args.dataset_dir
    )

    automm_hyperparameters['optimization.learning_rate'] = args.lr
    automm_hyperparameters['optimization.end_lr'] = args.lr

    tabular_hyperparameters = {
        'GBM': [
            {},
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
        ],
        'CAT': {},
        'XGB': {},
        'NN': {},
        'AG_AUTOMM_NN': automm_hyperparameters,
    }
    ag_args_ensemble = {
            '_disable_parallel_fitting': True
        }

    if args.mode == 'single':
        ### model initalization
        predictor = AutoMMPredictor(
            label=train_data.label_column,
            problem_type=train_data.problem_type,
            eval_metric=train_data.metric,
            path=args.exp_dir,
            verbosity=4,
        )

        ### model training
        predictor.fit(
            train_data=train_data.data,
            tuning_data=val_data.data,
            seed=args.seed,
            hyperparameters=automm_hyperparameters
        )

        ### model inference
        scores = predictor.evaluate(
            data=test_data.data,
            metrics=[test_data.metric]
        )
        with open(os.path.join(args.exp_dir, 'scores.json'), 'w') as f:
            json.dump(scores, f)
        print(scores)
    elif args.mode == 'weighted' or args.mode == 'single_bag5' or args.mode == 'stack5':
        if args.mode == 'single_bag5':
            tabular_hyperparameters = {
                'AG_AUTOMM_NN': automm_hyperparameters,
            }
            num_bag_folds, num_stack_levels = 5, 0
        elif args.mode == 'weighted':
            num_bag_folds, num_stack_levels = None, None
        elif args.mode == 'stack5':
            num_bag_folds, num_stack_levels = 5, 1
        else:
            raise NotImplementedError
        from autogluon.tabular import TabularPredictor
        predictor = TabularPredictor(eval_metric=train_data.metric,
                                     label=train_data.label_column,
                                     path=args.exp_dir)
        predictor.fit(train_data.data,
                      hyperparameters=tabular_hyperparameters,
                      num_bag_folds=num_bag_folds,
                      num_stack_levels=num_stack_levels,
                      ag_args_ensemble=ag_args_ensemble)
        leaderboard = predictor.leaderboard()
        leaderboard.to_csv(os.path.join(args.exp_dir, 'leaderboard.csv'))
    else:
        raise NotImplementedError
    scores = predictor.evaluate(
        data=test_data.data
    )
    with open(os.path.join(args.exp_dir, 'scores.json'), 'w') as f:
        json.dump(scores, f)
    print(scores)

    predictions = predictor.predict(
        data=test_data.data
    )
    predictions.to_csv(args.exp_dir, 'predictions.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ad', type=str)
    parser.add_argument('--dataset_dir', default='./dataset', type=str)
    parser.add_argument('--exp_dir', default=None, type=str)
    parser.add_argument('--lr', default=1e-04, type=float)
    parser.add_argument('--mode', choices=['single', 'weighted', 'single_bag5', 'stack5'], default='single')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    if args.exp_dir is None:
        args.exp_dir = f'./results/{args.dataset_name}'

    main(args)
