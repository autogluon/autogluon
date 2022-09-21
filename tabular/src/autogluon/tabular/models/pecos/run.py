# Run PECOS with AutoGluon. Testing file provided for convenience
from autogluon.tabular import TabularDataset
from pecos_model import PecosModel
import pandas as pd

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
label = 'class'  # specifies which column do we want to predict

print(train_data.head(5))

# Clean data

# Separate features and labels
X = train_data.drop(columns=[label])
y = train_data[label]
X_test = test_data.drop(columns=[label])
y_test = test_data[label]

from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
# Construct a LabelCleaner to neatly convert labels to float/integers during model training/inference, can also use to inverse_transform back to original.
problem_type = infer_problem_type(y=y)  # Infer problem type (or else specify directly)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_clean = label_cleaner.transform(y)

print(f'Labels cleaned: {label_cleaner.inv_map}')
print(f'inferred problem type as: {problem_type}')
print('Cleaned label values:')
print(y_clean.head(5))


# Clean features (i.e. strings)

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
set_logger_verbosity(2)  # Set logger so more detailed logging is shown for tutorial

feature_generator = AutoMLPipelineFeatureGenerator()
X_clean = feature_generator.fit_transform(X)

print(X_clean.head(5))

# Fit model

cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
text_features = None
num_features = ['age', 'fnlwgt', 'education-num', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week']

print(X)
kwargs = {'cat_features':cat_features, 'text_features': text_features, 'num_features': num_features, 'model_type': "XRLinear"}
custom_model = PecosModel(**kwargs)
custom_model.fit(X=X_clean, y=y_clean)  # Fit custom model

# To save to disk and load the model, do the following:
# load_path = custom_model.path
# custom_model.save()
# del custom_model
# custom_model = CustomRandomForestModel.load(path=load_path)

# Test

print(X_test.head(5))

X_test_clean = feature_generator.transform(X_test)

print(X_test_clean.head(5))

y_pred = custom_model.predict(X_test_clean)
y_pred = pd.Series(y_pred)
print(y_pred.head(5))

# Score model

y_test_clean = label_cleaner.transform(y_test)
y_pred_orig = label_cleaner.inverse_transform(y_pred)
print(y_pred_orig.head(5))


score = custom_model.score(X_test_clean, y_test_clean)
print(f'Test score ({custom_model.eval_metric.name}) = {score}')

run_bagged = False
if run_bagged:
    from autogluon.core.models import BaggedEnsembleModel
    bagged_custom_model = BaggedEnsembleModel(PecosModel())
    bagged_custom_model.params['fold_fitting_strategy'] = 'sequential_local'
    bagged_custom_model.fit(X=X_clean, y=y_clean, k_fold=10)  # Perform 10-fold bagging
    bagged_score = bagged_custom_model.score(X_test_clean, y_test_clean)
    print(f'Test score ({bagged_custom_model.eval_metric.name}) = {bagged_score} (bagged)')
    print(f'Bagging increased model accuracy by {round(bagged_score - score, 4) * 100}%!')


from autogluon.tabular import TabularPredictor

run_with_defined_hyperparameters = False
if run_with_defined_hyperparameters:
    kwargs = {'cat_features':cat_features, 'text_features': text_features, 'num_features': num_features}
    custom_hyperparameters = {PecosModel: [kwargs]}#[{'cat_features': cat_features, 'text_features': text_features, 'num_features': num_features}]}#, {'seed': 10}, {'seed': 20}]}
    predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters)
    predictor.leaderboard(test_data, silent=True)


from autogluon.core.space import Categorical, Int, Real, Bool
custom_hyperparameters_hpo = {PecosModel: {
    'cat_features': cat_features,
    'text_features': text_features,
    'num_features': num_features,
    'max_leaf_size': Int(lower=50, upper=200),
    'nr_splits': Categorical(2, 4, 8, 16, 32, 64, 128),
    'spherical': Bool(),
    'kmeans_max_iter': Int(lower=5, upper=100),
    'solver_type': Categorical("L2R_L2LOSS_SVC_DUAL", "L2R_L1LOSS_SVC_DUAL", "L2R_LR_DUAL", "L2R_L2LOSS_SVC_PRIMAL"),
    #'coefficient_positive': Real(lower=0.1, upper=1.0),
    #'coefficient_negative': Real(lower=0.1, upper=1.0),
    #'bias': Real(lower=0.5, upper=2.0),
    'negative_sampling': Categorical("tfn", "man", "tfn+man"),
    'sparsity_threshold': Real(lower=0.05, upper=0.2),
    }}
# Hyperparameter tune CustomRandomForestModel for 20 seconds
predictor = TabularPredictor(label=label).fit(train_data,
                                              hyperparameters=custom_hyperparameters_hpo,
                                              hyperparameter_tune_kwargs='auto',
                                              time_limit=20)
