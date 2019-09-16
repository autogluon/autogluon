import collections
import pickle
import re

import numpy as np

from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor
from tabular.sandbox.swallows.processors.utils import perform_replacements


class AmazonLinkProcessor(AbstractPreprocessor):

    def __init__(self, path, is_training, dataset_name, min_link_freq, domains_pickle_prefix=None):
        self.name = "AmazonLinkProcessor"
        self.is_training = is_training
        self.path = path
        self.dataset_name = dataset_name
        self.domains_pickle_prefix = self.dataset_name if domains_pickle_prefix is None else domains_pickle_prefix
        self.min_link_freq = min_link_freq

    @staticmethod
    def get_high_frequency_links(df, min_freq):
        domains = (','.join((df['amazon_domains'].str.lower().values.tolist()))).split(',')
        ctr = collections.Counter(domains)
        ctr
        print(ctr)
        freq = np.vstack([np.array(list(ctr.keys())), np.array(list(ctr.values()))])
        return list(freq[0][np.argwhere(freq[1].astype(int) >= min_freq).squeeze()])

    def run(self, context, df):
        print(f'Keeping only link with freq >= {self.min_link_freq}')
        df['details'] = df['details'].str.replace('http:\\\\{2}', 'http://')
        df['details'] = df['details'].str.replace('https:\\\\{2}', 'https://')
        matches = df['details'].str.extractall('http[s]?://(?P<amazon_domain>[a-z\\-.]+){1}?.amazon.com', flags=re.IGNORECASE)
        df['amazon_domains'] = matches.groupby(level=0)['amazon_domain'].apply(set).map(lambda x: ','.join(x))
        df['amazon_domains'] = df['amazon_domains'].fillna('')

        domains_pickle_location = self.path / f'{self.domains_pickle_prefix}_domains_to_add.pickle'
        if self.is_training:
            domains_to_add = self.get_high_frequency_links(df, self.min_link_freq)
            print(f'Going to add {len(domains_to_add)} domains as features')
            with open(domains_pickle_location, 'wb') as f:
                pickle.dump(domains_to_add, f)
            print(f'Train set - {domains_pickle_location} saved: {len(domains_to_add)} entries')
        else:
            with open(domains_pickle_location, 'rb') as f:
                domains_to_add = pickle.load(f)
            print(f'Test set - {domains_pickle_location} loaded: {len(domains_to_add)} entries')

        for domain in domains_to_add:
            if domain != '':
                col_name = 'amz_domain_' + re.sub('[^a-z]+', '_', domain)
                df[col_name] = df['amazon_domains'].str.contains(domain, flags=re.IGNORECASE)
                print(f'\t- {col_name}')

        print('Adding amazon domains tokens')

        perform_replacements(df, 'details', {f'http[s]?://{domain}.amazon.com[\\w/.#?=+&\\-%0-9]*': 'xx' + re.sub('[^a-z]+', '', domain) for domain in domains_to_add if domain != ''})

        return df
