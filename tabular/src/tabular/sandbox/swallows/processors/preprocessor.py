#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import os
import pickle

import pandas as pd

from tabular.sandbox.swallows.processors.constants import CTX_DF, DATASETS


class AbstractPreprocessor(object):
    name = None
    persist_path = None

    def runstage(self, context: dict):
        """Common stage code"""
        print("{} - Running {}...".format(self.get_timestamp(), self.name))

        out = self.run(context, context['df'])
        context['df'] = out
        print("{} ------------------------------------------------------------------------------- Completed {}".format(self.get_timestamp(), self.name))
        return out

    def run(self, context, df):
        raise NotImplementedError("Should have implemented this")

    @staticmethod
    def get_timestamp():
        return '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())

    def save_state(self):
        if self.persist_path is None:
            raise ValueError('Persist path not set for', type(self).__name__, 'object')
        if self.persist_path != '':
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        file = open(self.persist_path, 'wb')
        pickle.dump(self, file)
        print('Saved', type(self).__name__, 'object to', self.persist_path)

    @classmethod
    def load_state(cls, persist_path):
        file = open(persist_path, 'rb')
        object = pickle.load(file)
        if not isinstance(object, cls):
            raise TypeError('Loaded', type(object).__name__, 'object from', persist_path, 'but expected', cls.__name__)
        print('Loaded', type(object).__name__, 'object from', persist_path)
        return object


class AbstractStagePreprocessor(AbstractPreprocessor):

    def __init__(self, path, dataset_name, is_training, file_prefix_source, file_prefix_target):
        self.name = None
        self.path = path
        self.file_prefix_source = file_prefix_source
        self.file_prefix_target = file_prefix_target
        self.dataset_name = dataset_name
        self.is_training = is_training

    def get_processors(self):
        raise NotImplementedError("Please Implement get_processors method")

    def before_processing_hook(self, df):
        return df

    def run(self, context={}):
        input = self.load_stage_data()
        input = self.before_processing_hook(input)
        context[CTX_DF] = input

        out = None
        for stage in self.get_processors():
            out = stage.runstage(context)

        self.save_stage_data(out)

        return out

    def save_stage_data(self, out):
        out.to_parquet(self.path / f'{self.file_prefix_target}.parquet')
        if DATASETS[self.dataset_name]['write_csv']:
            cols_to_save = DATASETS[self.dataset_name]['csv_cols']
            out_csv = out[DATASETS[self.dataset_name]['csv_cols']] if cols_to_save is not None else out
            out_csv.to_csv(self.path / f'{self.file_prefix_target}.csv', index=False)
        print(f'Stage data is saved >> {self.dataset_name} | {DATASETS[self.dataset_name]}')

    def load_stage_data(self):
        df = pd.read_parquet(self.path / f'{self.file_prefix_source}.parquet')
        print(f'Stage data is loaded << {self.dataset_name} | {DATASETS[self.dataset_name]}')
        return df
