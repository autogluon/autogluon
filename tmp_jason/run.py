import argparse
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.models import CatBoostModel, KNNModel, LGBModel, XGBoostModel
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--train_path', help='train dataset path', type=str, default='dataset/australian/train_data.csv')
parser.add_argument('-g', '--test_path', help='train dataset path', type=str, default='dataset/australian/test_data.csv')
parser.add_argument('-m', '--mode', help='what AutoGluon setting to try', choices=['model', 'model-stack', 'ag', 'ag-stack'], default='model')
parser.add_argument('-n', '--num_resource', help='number of resource to allocate across all features', type=int, default=None)
parser.add_argument('-l', '--label', help='name of the label column', type=str, default='class')
parser.add_argument('-p', '--prune', help='to use fit_with_prune or not', dest='prune', action='store_true')
parser.add_argument('-r', '--ratio', help='what percentage of features to prune at once', type=float, default=0.05)
parser.add_argument('-x', '--fi', help='feature importance strategy', default='uniform', choices=['uniform', 'backwardsearch'])
parser.add_argument('-y', '--fp', help='feature pruning strategy or not', default='percentage', choices=['percentage', 'single'])
args = parser.parse_args()

train_data = pd.read_csv(args.train_path)  # .head(2500)
test_data = pd.read_csv(args.test_path)
X_test = test_data.drop(columns=[args.label])
y_test = test_data[args.label]


feature_prune_kwargs = {
    'fit_with_prune_kwargs': {
        'max_fits': 10,
        'stop_threshold': 1,
        'prune_ratio': args.ratio,
        'prune_threshold': 0.,
        'train_subsample_size': 50000,
        'fi_subsample_size': 5000,
        'min_fi_samples': 10000
    }
}

time_limit = 1800
if args.mode == 'model':
    presets = ['medium_quality_faster_train']
    custom_hyperparameters = {KNNModel: {}}
    extra_args = {}
elif args.mode == 'model-stack':
    presets = ['best_quality']
    # custom_hyperparameters = {KNNModel: {'ag_args': fit_with_prune_kwargs}}
    # custom_hyperparameters = {KNNModel: {}, CatBoostModel: {}, LGBModel: {}, XGBoostModel: {}}
    custom_hyperparameters = {LGBModel: {}}
    extra_args = {'num_bag_sets': 2, 'num_stack_levels': 1}
elif args.mode == 'ag':
    presets = ['medium_quality_faster_train']
    custom_hyperparameters = None
    extra_args = {}
else:
    presets = ['best_quality']
    custom_hyperparameters = None
    extra_args = {'num_bag_sets': 2, 'num_stack_levels': 1}

predictor = TabularPredictor(label=args.label)
if args.prune:
    predictor = predictor.fit(train_data, presets=presets, ag_args=feature_prune_kwargs, time_limit=time_limit, **extra_args,
                              ag_args_ensemble={'use_child_oof': False}, hyperparameters=custom_hyperparameters)
else:
    predictor = predictor.fit(train_data, presets=presets, **extra_args,
                              time_limit=time_limit, hyperparameters=custom_hyperparameters)

try:
    y_pred = predictor.predict(test_data)
    performance = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    print(performance)
except Exception as e:
    import pdb; pdb.post_mortem()
    print(e)
# import pdb; pdb.set_trace()
# result = predictor.feature_importance(data=test_data, model='KNeighbors_BAG_L2', features=['sex'])
# print(result)
# import pdb; pdb.set_trace()
# result = predictor.feature_importance(data=test_data, model='KNeighbors_BAG_L2', features=['sex'])
# print(result)
