""" Example script for defining and using feature generators in AutoGluon Tabular """

################
# Loading Data #
################

from autogluon import TabularPrediction as task

train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/train_data.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/test_data.csv')  # another Pandas DataFrame
label_column = 'class'  # specifies which column do we want to predict
sample_train_data = train_data.head(5)  # subsample for faster demo

# Separate features and labels
X = sample_train_data.drop(columns=[label_column])
y = sample_train_data[label_column]

X_test = test_data.drop(columns=[label_column])
y_test = test_data[label_column]

print(X)

##############################
# Fitting feature generators #
##############################

from autogluon.utils.tabular.features.generators import CategoryFeatureGenerator, IdentityFeatureGenerator

# IdentityFeatureGenerator is a 'do-nothing' feature generator if given default arguments. It will simply pass the data along.
identity_feature_generator = IdentityFeatureGenerator()

# fit_transform the generator using the input data. This must be done prior to calling transform.
X_transform = identity_feature_generator.fit_transform(X=X, verbosity=3)  # verbosity=3 to log more information during fit.
# identity_feature_generator.fit(X=X)  # This is identical to fit_transform, just without returning X_identity_out

# Because IdentityFeatureGenerator simply passes the data along, nothing changed.
assert X_transform.equals(X)

identity_feature_generator = IdentityFeatureGenerator(features_in=['age', 'workclass'])  # Limit the valid input to only 'age' and 'workclass' features.
X_transform = identity_feature_generator.fit_transform(X=X, verbosity=3)
print(X_transform)  # Now the output only contains the two features we declared in the input arguments to the generator, acting as a feature filter.

from autogluon.utils.tabular.features.feature_metadata import R_INT
identity_feature_generator = IdentityFeatureGenerator(infer_features_in_args={'valid_raw_types': [R_INT]}, verbosity=3)  # Limit the valid input to only integer features.
X_transform = identity_feature_generator.fit_transform(X=X)
print(X_transform)  # Now the output only contains the int type features, acting as a type filter.

# Our data contains object features at present, but this is not valid input to models, so lets convert them to category types.
category_feature_generator = CategoryFeatureGenerator(verbosity=3)
X_transform = category_feature_generator.fit_transform(X=X)
print(X_transform)  # Note that the int features were automatically filtered out of this output. This is due to the defaults of CategoryFeatureGenerator which does not handle features other than objects and categories.

#####################################
# Create a custom feature generator #
#####################################

from pandas import DataFrame
from autogluon.utils.tabular.features.generators import AbstractFeatureGenerator


# Feature generator to add 1 to all values of integer features.
class PlusOneFeatureGenerator(AbstractFeatureGenerator):
    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        # Here we can specify any logic we want to make a stateful feature generator based on the data.
        # Just call _transform since this isn't a stateful feature generator.
        X_out = self._transform(X)
        # return the output and the new special types of the data. For this generator, we don't add any new special types, so just return the input special types
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        # Here we can specify the logic taken to convert input data to output data post-fit. Here we can reference any variables created during fit if the generator is stateful.
        # Because this feature generator is not stateful, we simply add 1 to all features.
        return X + 1

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT])  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter.


################################
# Fit custom feature generator #
################################

plus_one_feature_generator = PlusOneFeatureGenerator(verbosity=3)
X_transform = plus_one_feature_generator.fit_transform(X=X)
print(X_transform)

##################################
# Multi-stage feature generators #
##################################

from autogluon.utils.tabular.features.generators import FillNaFeatureGenerator, AsTypeFeatureGenerator, BulkFeatureGenerator, DropUniqueFeatureGenerator

bulk_feature_generator = BulkFeatureGenerator(
    generators=[
        [AsTypeFeatureGenerator()],  # Stage 1: Convert feature types to be the same as during fit.
        [FillNaFeatureGenerator()],  # Stage 2: Fill NaN values of data
        [  # Stage 3: Add 1 to all int features and convert all object features to category features. Concatenate the outputs of each.
            PlusOneFeatureGenerator(),
            CategoryFeatureGenerator(),
        ],
        [DropUniqueFeatureGenerator()],  # Stage 4: Drop any features which are always the same value (useless).
    ],
    verbosity=3
)
X_transform = bulk_feature_generator.fit_transform(X=X)  # Fits each stage of the BulkFeatureGenerator sequentially, with inputs to Stage N+1 coming from the output of Stage N.
print(X_transform)
X_test_transform = bulk_feature_generator.transform(X=X_test)  # Can now transform the test data based on the fit data.
print(X_test_transform)

###############################
# Pre-made feature generators #
###############################

from autogluon.utils.tabular.features.generators import AutoMLPipelineFeatureGenerator

# This is the default feature generator of AutoGluon, and contains many stages of preprocessing made to handle many types of data.
# For most users, this should be all they need to get high quality features that are ready to fit models.
auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
X_transform = auto_ml_pipeline_feature_generator.fit_transform(X=X)
print(X_transform)
X_test_transform = auto_ml_pipeline_feature_generator.transform(X=X_test)
print(X_test_transform)
