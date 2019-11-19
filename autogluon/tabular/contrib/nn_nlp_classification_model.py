import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastai.basic_data import DatasetType
from fastai.basic_train import load_callback
from fastai.core import PathLikeOrBinaryStream, defaults, is_pathlike
from fastai.data_block import LabelLists, ItemList
from fastai.metrics import accuracy
from fastai.text import TextList, AWD_LSTM, text_classifier_learner, Vocab

from autogluon.tabular.ml.models.abstract.abstract_model import AbstractModel
from autogluon.tabular.contrib.nn_nlp_lm_model import VOCABULARY_PKL, ENCODER_FORWARD, ENCODER_BACKWARDS

LABEL = '__label__'


class NNNLPClassificationModel(AbstractModel):
    def __init__(self, path, name, params, problem_type, objective_func, debug=0, train_backwards=False):
        model = None
        super().__init__(path=path, name=name, model=model, problem_type=problem_type, objective_func=objective_func, debug=debug)
        self.params = params
        self.train_backwards = train_backwards
        self.metric = params['metric']

        # Batch size - use maximum possible as GPU allowing
        self.bs = params['bs']

        # Back propagation through time depth in LSTM; use multipy of 8 if fp16 training is used
        self.bptt = params['bptt']

        # Weight decay
        self.wd = params['wd']

        self.feature_field = params['feature_field']

        # Dropout multiplier (AWD LSTM regularization)
        self.drop_mult = params['drop_mult']

        self.encoders_pointer_location = params['encoders.pointer.location']

        self._model_internal = None

    def __getstate__(self):
        state = dict(self.__dict__)
        # Don't save internal state - it's un-pickle-able
        state['_model_internal'] = None
        return state

    def fit(self, X_train, Y_train, X_test=None, Y_test=None):
        if not os.path.exists(f'{self.path}/models/'):
            os.makedirs(f'{self.path}/models/')

        df_train, train_idx, val_idx = self._generate_datasets(X_train, Y_train, X_test, Y_test)

        print(f'Loading pointers from {self.encoders_pointer_location}')
        with open(self.encoders_pointer_location, 'r') as f:
            lm_path = f.read()
        print(f'Language model path is {lm_path}')

        vocab_path = f'{lm_path}{VOCABULARY_PKL}'
        lm_vocab: Vocab = pickle.load(open(vocab_path, "rb"))
        print(f'Loaded language model vocabulary: {len(lm_vocab.itos)} entries')

        if self.train_backwards:
            encoder_path = f'{lm_path}models/{ENCODER_BACKWARDS}.pth'
        else:
            encoder_path = f'{lm_path}models/{ENCODER_FORWARD}.pth'
        print(f'Using encoder: {encoder_path}')
        shutil.copyfile(encoder_path, f'{self.path}models/encoder.pth')

        metrics_map = {
            'accuracy': accuracy,
        }

        nn_metric = metrics_map[self.metric]

        print('Preparing data for classifier')
        data = (TextList.from_df(df_train[[self.feature_field, LABEL]], cols=self.feature_field, path=self.path, vocab=lm_vocab)
                .split_by_idxs(train_idx, val_idx)
                .label_from_df(cols=LABEL)
                .databunch(bs=self.bs, backwards=self.train_backwards))
        self.train_model(data, nn_metric)

    def train_model(self, data, nn_metric):
        model = text_classifier_learner(data, AWD_LSTM, drop_mult=self.drop_mult, metrics=[nn_metric], pretrained=False)
        model.load_encoder('encoder')
        model.fit_one_cycle(self.params['training.classifier.l0.epochs'], self.params['training.classifier.l0.lr'], moms=(0.8, 0.7))
        model.freeze_to(-2)
        model.fit_one_cycle(self.params['training.classifier.l1.epochs'], self.params['training.classifier.l1.lr'], moms=(0.8, 0.7))
        model.freeze_to(-3)
        model.fit_one_cycle(self.params['training.classifier.l2.epochs'], self.params['training.classifier.l2.lr'], moms=(0.8, 0.7))
        model.unfreeze()
        model.fit_one_cycle(self.params['training.classifier.l3.epochs'], self.params['training.classifier.l3.lr'], moms=(0.8, 0.7))
        model.save(self.name)
        model.export(f'models/{self.name}-export.pkl', destroy=True)
        model.destroy()

    def _generate_datasets(self, X_train, Y_train, X_test, Y_test):
        df_train = pd.concat([X_train.copy(), X_test.copy()]).reset_index(drop=True)
        df_train[LABEL] = pd.concat([Y_train.copy(), Y_test.copy()]).reset_index(drop=True)
        train_idx = np.arange(len(X_train))
        val_idx = np.arange(len(X_test)) + len(X_train)
        return df_train, train_idx, val_idx

    def predict_proba(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X)

        model = self._get_initialized_model(f'models/{self.name}-export.pkl', test=TextList.from_df(X, cols=self.feature_field), backwards=self.train_backwards)
        preds, _ = model.get_preds(ds_type=DatasetType.Test, ordered=True)
        preds = preds.numpy()

        # TODO: add other problem types cases
        return preds

    def _get_initialized_model(self, file: PathLikeOrBinaryStream, test: ItemList, **db_kwargs):
        if self._model_internal is None:
            self._model_internal = {}
            source = Path(self.path) / file if is_pathlike(file) else file
            state = torch.load(source, map_location='cpu') if defaults.device == torch.device('cpu') else torch.load(source)
            model = state.pop('model')
            src = LabelLists.load_state(self.path, state.pop('data'))

            data = src.databunch(**db_kwargs)
            self._model_internal['data'] = data

            cb_state = state.pop('cb_state')
            clas_func = state.pop('cls')
            inference_learner = clas_func(data, model, **state)
            inference_learner.callback_fns = state['callback_fns']
            inference_learner.callbacks = [load_callback(c, s, inference_learner) for c, s in cb_state.items()]
            self._model_internal['model'] = inference_learner

        self._model_internal['data'].add_test(test)
        return self._model_internal['model']
