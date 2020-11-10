
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

predictor1 = task.fit(
    train_data=train_data,
    label=label,
    hyperparameters=hyperparameters,
)
predictor1.leaderboard(test_data)

##################################
# Fitting with the new Predictor #
##################################

predictor2 = TabularPredictorV2(label=label)

# Showing that the new Predictor can be saved and loaded prior to calling .fit()
output_dir = predictor2.output_directory
predictor2.save()
del predictor2
predictor2 = TabularPredictorV2.load(output_directory=output_dir)

predictor2.fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
)
predictor2.leaderboard(test_data)

# Showing that the new Predictor can be reloaded after fit in the same fashion as the old Predictor
predictor2.save()
del predictor2
predictor2 = TabularPredictorV2.load(output_directory=output_dir)
predictor2.leaderboard(test_data)

#######################################################################
# Constructing old and new Predictor objects using the Learner object #
#######################################################################

learner = predictor2._learner

predictor_new_loaded_from_learner = TabularPredictorV2.from_learner(learner=learner)
predictor_new_loaded_from_learner.leaderboard(test_data)

predictor_old_loaded_from_learner = TabularPredictor(learner=learner)
predictor_old_loaded_from_learner.leaderboard(test_data)
