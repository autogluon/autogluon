import os
import pickle

import pandas as pd
import six
from fastai.basic_data import load_data
from fastai.text import TextList, AWD_LSTM, TokenizeProcessor, NumericalizeProcessor, language_model_learner

from autogluon.tabular.ml.constants import LANGUAGE_MODEL
from autogluon.tabular.ml.models.abstract.abstract_model import AbstractModel

LM_DATA_PKL = 'data_lm.pkl'
VOCABULARY_PKL = 'vocab.pkl'
ENCODER_FORWARD = 'fwd_enc'
ENCODER_BACKWARDS = 'bwd_enc'
LABEL = '__label__'


class NNNLPLanguageModel(AbstractModel):
    def __init__(self, path, name, params, debug=0, train_backwards=True):
        model = None
        super().__init__(path=path, name=name, model=model, problem_type=LANGUAGE_MODEL, objective_func=None, debug=debug)
        self.params = params
        self.train_backwards = train_backwards

        # Batch size - use maximum possible as GPU allowing
        self.bs = params['bs']

        # Back propagation through time depth in LSTM; use multipy of 8 if fp16 training is used
        self.bptt = params['bptt']

        self.test_split_ratio_pct = params['test_split_ratio_pct']

        # Weight decay
        self.wd = params['wd']

        # Maximum size of vocabulary
        self.max_vocab = params['max_vocab']

        self.feature_field = params['feature_field']

        # Dropout multiplier (AWD LSTM regularization)
        self.drop_mult = params['drop_mult']

        self.test_split_ratio_pct = params['test_split_ratio_pct']

        self.pretraining_epochs = params['training.lm.pretraining.epochs']
        self.pretraining_lr = params['training.lm.pretraining.lr']
        self.finetune_epochs = params['training.lm.finetune.epochs']
        self.finetune_lr = params['training.lm.finetune.lr']

        self.encoders_pointer_location = params['encoders.pointer.location']

    def predict(self, X, preprocess=True):
        raise NotImplementedError('This model is not supposed to be used for predict')

    def predict_proba(self, X, preprocess=True):
        raise NotImplementedError('This model is not supposed to be used for predict')

    def fit(self, X_train, Y_train, X_test=None, Y_test=None):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # TODO: review if we can avoid X_train/X_test here
        df_train = self._generate_datasets(X_train, X_test)

        print('Packaging data for training...')
        txt_proc = [TokenizeProcessor(), NumericalizeProcessor(vocab=None, max_vocab=self.max_vocab)]
        data_lm = (TextList.from_df(df_train, path=self.path, processor=txt_proc)
                   .split_by_rand_pct(self.test_split_ratio_pct)
                   .label_for_lm()
                   .databunch(bs=self.bs, bptt=self.bptt))
        data_lm.save(LM_DATA_PKL)

        print('Training forward language model')
        learn = self.train_lm_model(data_lm)
        learn.save_encoder(ENCODER_FORWARD)
        learn.destroy()

        if self.train_backwards:
            print('Training backward language model')
            data_bwd = load_data(self.path, LM_DATA_PKL, bs=self.bs, bptt=self.bptt, backwards=True)
            learn = self.train_lm_model(data_bwd)
            learn.save_encoder(ENCODER_BACKWARDS)
            learn.destroy()

        # Write vocabulary, so other models can read it
        vocab_path = f'{self.path}{VOCABULARY_PKL}'
        with open(vocab_path, 'wb') as f:
            pickle.dump(data_lm.vocab, f)
        print(f'recorded vocab to {vocab_path}')

        # Write pointer to encoders location so other models can read it
        with open(self.encoders_pointer_location, 'wb') as f:
            f.write(six.b(self.path))
        print(f'recorded lm models pointer to {self.encoders_pointer_location} -> {self.path}')

    def train_lm_model(self, data_lm):
        learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=self.drop_mult)
        learn.fit_one_cycle(self.pretraining_epochs, self.pretraining_lr, moms=(0.8, 0.7), wd=self.wd)
        learn.unfreeze()
        learn.fit_one_cycle(self.finetune_epochs, self.finetune_lr, moms=(0.8, 0.7), wd=self.wd)
        return learn

    def _generate_datasets(self, X_train, X_test):
        return pd.concat([X_train[[self.feature_field]], X_test[[self.feature_field]]]).reset_index(drop=True)
