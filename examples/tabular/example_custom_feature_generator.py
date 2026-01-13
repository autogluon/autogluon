"""
Example script for defining and using FeatureGenerators in AutoGluon Tabular.
FeatureGenerators act to clean and prepare the data to maximize predictive accuracy in downstream models.
FeatureGenerators are stateful data preprocessors which take input data (pandas DataFrame) and output transformed data (pandas DataFrame).
FeatureGenerators are first fit on training data through the .fit_transform() function, and then transform new data through the .transform() function.
These generators can do anything from filling NaN values (FillNaFeatureGenerator), dropping duplicate features (DropDuplicatesFeatureGenerator), generating ngram features from text (TextNgramFeatureGenerator), and much more.
In AutoGluon's TabularPredictor, the input data is transformed via a FeatureGenerator before entering a machine learning model. Some models use this transformed input directly and others perform further transformations before making predictions.

This example is intended for advanced users that have a strong understanding of feature engineering and data preparation.
Most users can get strong performance without specifying custom feature generators due to the generic and powerful default feature generator used by AutoGluon.
An advanced user may wish to create a custom feature generator to:
    1. Experiment with different preprocessing pipelines to improve model quality.
    2. Have full control over what data is being sent to downstream models.
    3. Migrate existing pipelines into AutoGluon for ease of use and deployment.
    4. Contribute new feature generators to AutoGluon.
"""

################
# Loading Data #
################

from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset(
    "https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/train_data.csv"
)  # can be local CSV file as well, returns Pandas DataFrame
test_data = TabularDataset(
    "https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification/test_data.csv"
)  # another Pandas DataFrame
label = "class"  # specifies which column do we want to predict
sample_train_data = train_data.head(100)  # subsample for faster demo

# Separate features and labels
# Make sure to not include your label/target column when sending input to the feature generators, or else the label will be transformed as well.
X = sample_train_data.drop(columns=[label])
y = sample_train_data[label]

X_test = test_data.drop(columns=[label])
y_test = test_data[label]

print(X)

##############################
# Fitting feature generators #
##############################

from autogluon.features.generators import CategoryFeatureGenerator, IdentityFeatureGenerator

# IdentityFeatureGenerator is a 'do-nothing' feature generator if given default arguments. It will simply pass the data along.
identity_feature_generator = IdentityFeatureGenerator()

# fit_transform the generator using the input data. This must be done prior to calling transform.
X_transform = identity_feature_generator.fit_transform(
    X=X, verbosity=3
)  # verbosity=3 to log more information during fit.
# identity_feature_generator.fit(X=X)  # This is identical to fit_transform, just without returning X_identity_out

# Because IdentityFeatureGenerator simply passes the data along, nothing changed.
assert X_transform.equals(X)

identity_feature_generator = IdentityFeatureGenerator(
    features_in=["age", "workclass"]
)  # Limit the valid input to only 'age' and 'workclass' features.
X_transform = identity_feature_generator.fit_transform(X=X, verbosity=3)
print(
    X_transform.head(5)
)  # Now the output only contains the two features we declared in the input arguments to the generator, acting as a feature filter.

from autogluon.common.features.types import R_INT

identity_feature_generator = IdentityFeatureGenerator(
    infer_features_in_args={"valid_raw_types": [R_INT]}, verbosity=3
)  # Limit the valid input to only integer features.
X_transform = identity_feature_generator.fit_transform(X=X)
print(X_transform.head(5))  # Now the output only contains the int type features, acting as a type filter.

# Our data contains object features at present, but this is not valid input to models, so lets convert them to category types.
category_feature_generator = CategoryFeatureGenerator(verbosity=3)
X_transform = category_feature_generator.fit_transform(X=X)
print(
    X_transform.head(5)
)  # Note that the int features were automatically filtered out of this output. This is due to the defaults of CategoryFeatureGenerator which does not handle features other than objects and categories.

#####################################
# Create a custom feature generator #
#####################################

from pandas import DataFrame

from autogluon.features.generators import AbstractFeatureGenerator


# Feature generator to add k to all values of integer features.
class PlusKFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        # Here we can specify any logic we want to make a stateful feature generator based on the data.
        # Just call _transform since this isn't a stateful feature generator.
        X_out = self._transform(X)
        # return the output and the new special types of the data. For this generator, we don't add any new special types, so just return the input special types
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        # Here we can specify the logic taken to convert input data to output data post-fit. Here we can reference any variables created during fit if the generator is stateful.
        # Because this feature generator is not stateful, we simply add k to all features.
        return X + self.k

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(
            valid_raw_types=[R_INT]
        )  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter.


################################
# Fit custom feature generator #
################################

plus_three_feature_generator = PlusKFeatureGenerator(k=3, verbosity=3)
X_transform = plus_three_feature_generator.fit_transform(X=X)
print(X_transform.head(5))

##################################
# Multi-stage feature generators #
##################################

from autogluon.features.generators import (
    AsTypeFeatureGenerator,
    BulkFeatureGenerator,
    DropUniqueFeatureGenerator,
    FillNaFeatureGenerator,
    PipelineFeatureGenerator,
)

# BulkFeatureGenerator is an implementation of AbstractFeatureGenerator that allows for advanced multi-stage feature generation.
bulk_feature_generator = BulkFeatureGenerator(
    generators=[
        [AsTypeFeatureGenerator()],  # Stage 1: Convert feature types to be the same as during fit.
        [FillNaFeatureGenerator()],  # Stage 2: Fill NaN values of data
        [  # Stage 3: Add 5 to all int features and convert all object features to category features. Concatenate the outputs of each.
            PlusKFeatureGenerator(k=5),
            CategoryFeatureGenerator(),
        ],
        [DropUniqueFeatureGenerator()],  # Stage 4: Drop any features which are always the same value (useless).
    ],
    verbosity=3,
)
X_transform = bulk_feature_generator.fit_transform(
    X=X
)  # Fits each stage of the BulkFeatureGenerator sequentially, with inputs to Stage N+1 coming from the output of Stage N.
print(X_transform.head(5))
X_test_transform = bulk_feature_generator.transform(X=X_test)  # Can now transform the test data based on the fit data.
print(X_test_transform.head(5))

# PipelineFeatureGenerator is an implementation of BulkFeatureGenerator which automatically handles very common necessary stages without requiring them to be defined by the user.
# It is recommended that users who wish to create custom feature generators as input to predictor.fit() should base their generators off of PipelineFeatureGenerator as a best practice.
# The following results in the exact same final functionality as bulk_feature_generator, plus many additional edge case handling functionality:
pipeline_feature_generator = PipelineFeatureGenerator(
    generators=[
        # Stage 1: Convert feature types to be the same as during fit. Does not need to be specified.
        # Stage 2: Fill NaN values of data. Does not need to be specified.
        [  # Stage 3: Add 5 to all int features and convert all object features to category features. Concatenate the outputs of each.
            PlusKFeatureGenerator(k=5),
            CategoryFeatureGenerator(),
        ],
        # Stage 4: Drop any features which are always the same value (useless). Does not need to be specified.
    ],
    verbosity=3,
)
X_transform = pipeline_feature_generator.fit_transform(X=X)
print(X_transform.head(5))
X_test_transform = pipeline_feature_generator.transform(X=X_test)
print(X_test_transform.head(5))

###############################
# Pre-made feature generators #
###############################

from autogluon.features.generators import AutoMLPipelineFeatureGenerator

# This is the default feature generator of AutoGluon, and contains many stages of preprocessing made to handle many types of data.
# AutoMLPipelineFeatureGenerator is an implementation of PipelineFeatureGenerator
# For most users, this should be all they need to get high quality features that are ready to fit models.
auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
X_transform = auto_ml_pipeline_feature_generator.fit_transform(X=X)
print(X_transform.head(5))
X_test_transform = auto_ml_pipeline_feature_generator.transform(X=X_test)
print(X_test_transform.head(5))

###########################################################
# Specifying custom feature generator to TabularPredictor #
###########################################################

example_models = {"GBM": {}, "CAT": {}}
example_models_2 = {"RF": {}, "KNN": {}}

# Because auto_ml_pipeline_feature_generator is already fit, it doesn't need to be fit again in predictor. Instead, train_data is just transformed by auto_ml_pipeline_feature_generator.transform(train_data).
# This allows the feature transformation to be completely independent of the training data, we could have used a completely different data source to fit the generator.
predictor = TabularPredictor(label="class").fit(
    train_data, hyperparameters=example_models, feature_generator=auto_ml_pipeline_feature_generator
)
X_test_transform_2 = predictor.transform_features(
    X_test
)  # This is the same as calling auto_ml_pipeline_feature_generator.transform(X_test)
assert X_test_transform.equals(X_test_transform_2)
# The feature metadata of the feature generator is also preserved. All downstream models will get this feature metadata information to make decisions on how they use the data.
assert predictor.feature_metadata.to_dict() == auto_ml_pipeline_feature_generator.feature_metadata.to_dict()
predictor.leaderboard(test_data)

# We can train multiple predictors with the same pre-fit feature generator. This can save a lot of time during experimentation if the fitting of the generator is expensive.
predictor_2 = TabularPredictor(label="class").fit(
    train_data, hyperparameters=example_models_2, feature_generator=auto_ml_pipeline_feature_generator
)
predictor_2.leaderboard(test_data)

# We can even specify our custom generator too (although it needs to do a bit more to actually improve the scores, in most situations just use AutoMLPipelineFeatureGenerator)
predictor_3 = TabularPredictor(label="class").fit(
    train_data, hyperparameters=example_models, feature_generator=plus_three_feature_generator
)
predictor_3.leaderboard(test_data)
