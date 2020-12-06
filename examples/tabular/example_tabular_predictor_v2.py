
from autogluon.tabular import TabularPrediction as task
from autogluon.tabular.task.tabular_prediction.predictor import TabularPredictor
from autogluon.tabular.task.tabular_prediction.predictor_v2 import TabularPredictorV2


################
# Loading data #
################

train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
label = 'class'
hyperparameters = {'RF': {}}
train_data = train_data.head(1000)  # subsample for faster demo

##################################
# Fitting with the old Predictor #
##################################

predictor1 = task.fit(train_data=train_data, label=label, hyperparameters=hyperparameters, num_bagging_folds=2)
predictor1.leaderboard(test_data)

##################################
# Fitting with the new Predictor #
##################################

predictor2 = TabularPredictorV2(label=label)

# Showing that the new Predictor can be saved and loaded prior to calling .fit()
output_dir = predictor2.output_directory
predictor2.save()
del predictor2
predictor2 = TabularPredictorV2.load(output_dir)

predictor2.fit_feature_generator(data=train_data)  # Can also fit the feature generator prior to calling fit
# train_data_transformed = predictor2.transform_features(train_data)  # FIXME: Make this work
predictor2.fit(train_data, hyperparameters=hyperparameters, num_bagging_folds=2)
predictor2.leaderboard(test_data)

# Showing that the new Predictor can be reloaded after fit in the same fashion as the old Predictor
predictor2.save()
del predictor2
predictor2 = TabularPredictorV2.load(output_dir)
predictor2.leaderboard(test_data)

####################################
# Advanced fit_extra functionality #
####################################

# Fit extra models at level 0, with 30 second time limit
hyperparameters_extra1 = {'GBM': {}, 'NN': {}}
predictor2.fit_extra(hyperparameters_extra1, time_limits=30)

# Fit new level 1 stacker models that use the level 0 models from the original fit and the previous fit_extra call as base models
hyperparameters_extra2 = {'CAT': {}, 'NN': {}}
base_model_names = predictor2.get_model_names(stack_name='core', level=0)
predictor2.fit_extra(hyperparameters_extra2, base_model_names=base_model_names)

# Fit a new 3-layer stack ensemble on top of level 1 stacker models
hyperparameters_extra3 = {
    0: {'XT': {}},
    1: {'NN': {}, 'RF': {}},
    2: {'XGB': {}, 'custom': ['GBM']}
}
base_model_names = predictor2.get_model_names(stack_name='core', level=1)
predictor2.fit_extra(hyperparameters_extra3, base_model_names=base_model_names)

predictor2.leaderboard(test_data)
