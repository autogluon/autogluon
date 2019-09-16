import pickle

import pandas as pd
from target_encoding import TargetEncoder

from tabular.sandbox.swallows.processors.constants import *
from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor


class TargetMeanEncoder(AbstractPreprocessor):

    def __init__(self, path, is_training, dataset_name, encoder_pickle_prefix=None):
        self.name = "TargetMeanEncoder"
        self.path = path
        self.is_training = is_training
        self.dataset_name = dataset_name
        self.encoder_pickle_prefix = self.dataset_name if encoder_pickle_prefix is None else encoder_pickle_prefix

    def run(self, context, df):
        FEATURES_TO_ENCODE = [
            'assigned_to_group',
            'case_type',
            'category',
            'channel rt',
            'first_assigned_group',
            'impact',
            'item',
            'site',
            'site_building_type',
            'site_city',
            'site_country',
            'site_loc_bldg_cd',
            'site_region',
            'site_state',
            'type',
            'lang',
            'policy_number',
            'policy_section',
            'amazon_domains',
            'wfss_workflow',
            'host_status',
        ]

        encoder_pickle_location = self.path / f'{self.encoder_pickle_prefix}_target_mean_encoder.pickle'
        xs = [df[feature].copy().astype('category').cat.codes.astype(float).values for feature in FEATURES_TO_ENCODE]

        if self.is_training:
            # https://github.com/KirillTushin/target_encoding
            enc = TargetEncoder()
            y = df['root_cause'].copy().astype('category').cat.codes
            mean_encoded_X = enc.transform_train(X=np.vstack(xs).T, y=y)

            with open(encoder_pickle_location, 'wb') as f:
                pickle.dump(enc, f)
            print(f'Train set - mean encoder saved: {encoder_pickle_location}')

        else:
            with open(encoder_pickle_location, 'rb') as f:
                enc = pickle.load(f)
            print(f'Test set - mean encoder loaded: {encoder_pickle_location}')
            mean_encoded_X = enc.transform_test(X=np.vstack(xs).T)

        df_new = pd.DataFrame(mean_encoded_X, columns=[f'{feature}_mean_encoded' for feature in FEATURES_TO_ENCODE])

        return df.join(df_new)
