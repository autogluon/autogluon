from autogluon.text.automm import AutoMMPredictor
import argparse
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

hyperparameters = {
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
    'optimization.max_epochs': 1000,
    'optimization.weight_decay': 1.0e-5,
    'optimization.lr_choice': None,
    'optimization.lr_schedule': "polynomial_decay",
    'optimization.warmup_steps': 0.,
    'optimization.patience': 16,
    'optimization.top_k': 1,
    'data.categorical.convert_to_text': False,
}

def main(args):
    assert args.dataset_name in TABULAR_DATASETS.keys(), 'Unsupported dataset name.'
    
    ### Dataset loading
    train_data = TABULAR_DATASETS[args.dataset_name](
        'train',args.dataset_dir
    )
    
    val_data = TABULAR_DATASETS[args.dataset_name](
        'val',args.dataset_dir
    )
    
    test_data = TABULAR_DATASETS[args.dataset_name](
        'test',args.dataset_dir
    )
    
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
        hyperparameters=hyperparameters
    )

    ### model inference
    predictor.evaluate(
        data=test_data.data,
        metrics=[test_data.metric]
    )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ad', type=str)
    parser.add_argument('--dataset_dir', default='/home/ubuntu/dataset', type=str)
    parser.add_argument('--exp_dir', default='/home/ubuntu/result/test', type=str)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    main(args)
