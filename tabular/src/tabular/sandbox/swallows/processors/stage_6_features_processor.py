import pickle

from tabular.sandbox.swallows.processors.feature_generator_light import FeatureGeneratorLight
from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor


class Stage6FeaturesGenerator(AbstractPreprocessor):

    def __init__(self, path, is_training, dataset_name, encoder_pickle_prefix=None):
        self.name = "Stage6FeaturesGenerator"
        self.path = path
        self.is_training = is_training
        self.dataset_name = dataset_name
        self.encoder_pickle_prefix = self.dataset_name if encoder_pickle_prefix is None else encoder_pickle_prefix

    def run(self, context, df):
        FeatureGeneratorLight()
        encoder_pickle_location = self.path / f'{self.encoder_pickle_prefix}_text_features_encoder.pickle'

        if self.is_training:
            enc = FeatureGeneratorLight()
            df_new = enc.fit_transform(df)
            enc.save_self(encoder_pickle_location)
            print(f'Train set - text features encoder saved: {encoder_pickle_location}')

        else:
            with open(encoder_pickle_location, 'rb') as f:
                enc = pickle.load(f)
            print(f'Test set - text features encoder loaded: {encoder_pickle_location}')
            df_new = enc.transform(df)

        return df_new
