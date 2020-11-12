# Credits for code in this script to Milan Cvitkovic,
# Xin Huang, Ashish Khetan and Zohar Karnin

import calendar
import datetime
import re
from collections import Counter
from datetime import datetime, date
from functools import partial
from typing import List, Union, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer, KBinsDiscretizer


class WontEncodeError(Exception):
    pass


class EncBase:
    cat_cards = []
    cont_dim = 0

    @property
    def cat_dim(self):
        return len(self.cat_cards)

    def clean_data(self, data, dtype=None) -> list:
        if isinstance(data, pd.Series):
            data = data.replace({np.nan: None}).to_list()
            if dtype == 'float':
                unclean_data = data
                data = []
                for i in unclean_data:
                    try:
                        data.append(float(i))
                    except (ValueError, TypeError):
                        data.append(None)
            return data

    def fit(self, data: pd.Series, dtype=None):
        """
        If dtype == 'float', clean_data will cast the contents of data to floats
        """
        if len(pd.unique(data)) == 1:
            raise WontEncodeError('Column contains only one value')
        data = self.clean_data(data, dtype)
        return data

    def enc_cat(self, data: Iterable):
        raise NotImplementedError

    def enc_cont(self, data: Iterable):
        raise NotImplementedError


class NullEnc(EncBase):
    """
    When you want to ignore a feature
    """

    def fit(self, data: Iterable):
        pass

    def enc_cat(self, data: Iterable):
        pass

    def enc_cont(self, data: Iterable):
        pass


class CategoricalOrdinalEnc(EncBase):
    def __init__(self, sorted_items=None):
        if sorted_items is not None:
            assert sorted_items == sorted(sorted_items)
            self.init_with_sorted_values(sorted_items)

    def fit(self, data: pd.Series):
        data = super().fit(data)
        distinct_vals = [i for i in pd.unique(data) if i is not None]
        sorted_vals = sorted(distinct_vals)
        if len(sorted_vals) >= 0.5 * len(data) or max(Counter(data).values()) < 10:  # sorry for hardcoding this...
            raise WontEncodeError("Too many unique values to bother embedding")
        self.init_with_sorted_values(sorted_vals)

    def init_with_sorted_values(self, sorted_vals):
        self._item_to_idx = {item: idx for idx, item in enumerate(sorted_vals, start=2)}
        self._item_to_idx[None] = 1
        self._item_to_idx[np.nan] = 1
        self.cat_cards = [len(set(self._item_to_idx.values()))]

    def enc_cat(self, data):
        """
        Values that the encoder has never seen before are returned as 1.  0 is reserved for padding.
        """
        data = self.clean_data(data)
        idxs = [self._item_to_idx.get(item, 1) for item in data]
        return torch.LongTensor(idxs).unsqueeze(1)

    def enc_cont(self, data):
        pass


class ScalarQuantileOrdinalEnc(EncBase):
    def __init__(self, n_bins_=None, bin_edges_=None):
        if n_bins_ is not None and bin_edges_ is not None:
            self.disc = self.get_new_base_enc()
            self.disc.n_bins_ = np.array([n_bins_])
            self.disc.bin_edges_ = np.array([np.array(bin_edges_), np.array(bin_edges_[:-1])])[
                                   :1]  # Dumb hack, but it's what sklearn needs
            self.cat_cards = [n_bins_ + 1]

    def fit(self, data):

        data = super().fit(data, dtype='float')
        fit_data = [i for i in data if i is not None]
        fit_data = np.array(fit_data).reshape(-1, 1)
        self.disc = self.get_new_base_enc()
        self.disc.fit(fit_data)
        self.cat_cards = [self.disc.n_bins_.item() + 1]

    def enc_cat(self, data):
        """
        Missing values are returned as category 1.  0 is reserved for padding.
        """
        data = self.clean_data(data, dtype='float')
        data = np.array(data).reshape(-1, 1)
        if None in data:
            idxs = np.full(len(data), -1, dtype=np.int)
            null_idxs = np.where(data == None)[0]
            val_idxs = np.where(data != None)[0]
            if len(val_idxs) > 0:
                vals = self.disc.transform(data[val_idxs]).reshape(-1)
                idxs[val_idxs] = vals + 2
            idxs[null_idxs] = 1
        else:
            idxs = self.disc.transform(data).reshape(-1) + 2
        return torch.LongTensor(idxs).unsqueeze(1)

    def enc_cont(self, data):
        pass

    @staticmethod
    def get_new_base_enc():
        return KBinsDiscretizer(n_bins=8,
                                encode='ordinal',
                                strategy='quantile')

    def get_base_enc_params(self):
        return self.disc.n_bins_, self.disc.bin_edges_


class ScalarRescaleEnc(EncBase):
    cont_dim = 2
    scaler = None

    def enc_cat(self, data):
        pass

    def enc_cont(self, scalars):
        """
        Returns len(scalars) x 2 tensor, where the second column is a one-hot flag for missing data values
        """
        scalars = self.clean_data(scalars, dtype='float')
        null_flag = np.full(len(scalars), np.nan, dtype=np.float32)
        vals = np.full(len(scalars), np.nan, dtype=np.float32)
        null_idxs = np.where(np.array(scalars) == None)[0]
        val_idxs = np.where(np.array(scalars) != None)[0]

        # One-hot flag for missing values
        null_flag[null_idxs] = 1
        null_flag[val_idxs] = 0
        null_flag = null_flag.reshape(-1, 1)

        # Transform scalar values
        vals[val_idxs] = np.array(scalars, dtype=np.float32)[val_idxs]
        vals = vals.reshape(-1, 1)
        vals = self.scaler.transform(vals) + 1e-7  # Extra 1e-7 to help with correctness testing
        vals[null_idxs] = 0

        encoded = np.hstack((vals, null_flag))
        encoded = encoded.clip(-5, 5)  # Guarding against outlier values
        return torch.FloatTensor(encoded)

    @staticmethod
    def get_new_base_enc():
        raise NotImplementedError

    def get_base_enc_params(self):
        raise NotImplementedError


class ScalarRobustScalerEnc(ScalarRescaleEnc):
    def __init__(self, center_=None, scale_=None):
        if center_ is not None and scale_ is not None:
            self.scaler = self.get_new_base_enc()
            self.scaler.center_ = center_
            self.scaler.scale_ = scale_

    def fit(self, data: pd.Series):
        data = super().fit(data, dtype='float')
        data = np.array(data).reshape(-1, 1)
        self.scaler = self.get_new_base_enc()
        self.scaler.fit(data)
        if any(sum(np.isnan(p) for p in self.get_base_enc_params())):
            self.scaler.center_ = 0
            self.scaler.scale_ = 1

    @staticmethod
    def get_new_base_enc():
        return RobustScaler()

    def get_base_enc_params(self):
        return self.scaler.center_, self.scaler.scale_


class ScalarPowerTransformerEnc(ScalarRescaleEnc):
    def __init__(self, lambdas_=None, scale_=None, mean_=None, var_=None, n_samples_seen_=None):
        if all(a is not None for a in [lambdas_, scale_, mean_, var_, n_samples_seen_]):
            self.scaler = self.get_new_base_enc()
            self.scaler.fit(
                [[0.0]])  # This is just to make the PowerTransformer initialize before we overwrite its params
            self.scaler.lambdas_ = np.array([lambdas_])
            self.scaler._scaler.scale_ = np.array([scale_])
            self.scaler._scaler.mean_ = np.array([mean_])
            self.scaler._scaler.var_ = np.array([var_])
            self.scaler._scaler.n_samples_seen_ = n_samples_seen_

    def fit(self, data):
        data = super().fit(data, dtype='float')
        data = np.array(data).reshape(-1, 1)
        self.scaler = self.get_new_base_enc()
        self.scaler.fit(data)

    @staticmethod
    def get_new_base_enc():
        return PowerTransformer(method='yeo-johnson', standardize=True, copy=True)

    def get_base_enc_params(self):
        return self.scaler.lambdas_, self.scaler._scaler.scale_, self.scaler._scaler.mean_, self.scaler._scaler.var_, self.scaler._scaler.n_samples_seen_


class ScalarQuantileTransformerEnc(ScalarRescaleEnc):
    def __init__(self, n_quantiles_=None, quantiles_=None, references_=None):
        if all(a is not None for a in [n_quantiles_, quantiles_, references_]):
            self.scaler = self.get_new_base_enc()
            self.scaler.n_quantiles_ = n_quantiles_
            self.scaler.quantiles_ = np.array(quantiles_).reshape(-1, 1)
            self.scaler.references_ = np.array(references_)

    def fit(self, data):
        data = super().fit(data, dtype='float')
        data = np.array(data).reshape(-1, 1)
        self.scaler = self.get_new_base_enc()
        self.scaler.fit(data)

    @staticmethod
    def get_new_base_enc():
        return QuantileTransformer()

    def get_base_enc_params(self):
        return self.scaler.n_quantiles_, self.scaler.quantiles_, self.scaler.references_


class DatetimeScalarEnc(EncBase):
    # int for type refers to the cardinality of the one-hot
    cols_types = [
        ('Year', 'float'),
        ('Month', 12),
        ('Week', 53),
        ('Day', 31),
        ('Dayofweek', 7),
        ('Dayofyear', 'float'),
        ('Is_month_end', 2),
        ('Is_month_start', 2),
        ('Is_quarter_end', 2),
        ('Is_quarter_start', 2),
        ('Is_year_end', 2),
        ('Is_year_start', 2),
        ('weekday_cos', 'float'),
        ('weekday_sin', 'float'),
        ('day_month_cos', 'float'),
        ('day_month_sin', 'float'),
        ('month_year_cos', 'float'),
        ('month_year_sin', 'float'),
        ('day_year_cos', 'float'),
        ('day_year_sin', 'float'),
    ]
    cont_dim = sum([n if type(n) == int else 1 for _, n in cols_types])

    def enc_cat(self, data):
        pass

    def enc_cont(self, datetimes):
        datetimes = self.clean_data(datetimes)
        df = pd.DataFrame({'dt': datetimes})
        add_datepart(df, field_name='dt', prefix='', drop=False)
        df = add_cyclic_datepart(df, field_name='dt', prefix='', drop=False)
        enc = torch.empty(len(datetimes), self.cont_dim)
        feats_done = 0
        for c, t in self.cols_types:
            feats_doing = (1 if t == 'float' else t)
            if t == 'float':
                feats = torch.FloatTensor(df[c].to_numpy()).view(-1, 1)
                if c == 'Year':
                    feats = (feats - 2000) / 10
                elif c == 'Dayofyear':
                    feats /= 365
            else:
                feats = torch.LongTensor(df[c].to_numpy().astype('int32')).view(-1, 1)
                if c in ['Month', 'Week', 'Day']:
                    feats -= 1
                feats = one_hot(feats, t)
            enc[:, feats_done: feats_done + feats_doing] = feats
            feats_done += feats_doing
        return enc


class DatetimeOrdinalEnc(EncBase):
    # These are all 1 larger than you'd expect to support missing values
    cols_types = [
        ('Month', 13),
        ('Week', 54),
        ('Day', 32),
        ('Dayofweek', 8),
        ('Is_month_end', 3),
        ('Is_month_start', 3),
        ('Is_quarter_end', 3),
        ('Is_quarter_start', 3),
        ('Is_year_end', 3),
        ('Is_year_start', 3)
    ]
    cat_cards = [n for _, n in cols_types]

    def enc_cat(self, datetimes):
        # todo: add support for missing values, which should get encoded as 1.
        datetimes = self.clean_data(datetimes)
        df = pd.DataFrame({'dt': datetimes})
        add_datepart(df, field_name='dt', prefix='', drop=False)
        feats = []
        for c, t in self.cols_types:
            f = torch.LongTensor(df[c].to_numpy().astype('int32'))
            if c in ['Month', 'Week', 'Day']:
                f -= 1
            feats.append(f)
        feats = torch.stack(feats, dim=1) + 2  # + 2 for missing and padding
        return feats

    def enc_cont(self, data):
        pass


class LatLongScalarEnc(EncBase):
    cont_dim = 5

    def enc_cat(self, data):
        pass

    def enc_cont(self, latlongs):
        latlongs = self.clean_data(latlongs)
        if isinstance(latlongs[0], str):
            fixed = []
            for ll in latlongs:
                lat, long = ll.strip('()').split(',')
                lat, long = float(lat), float(long)
                fixed.append((lat, long))
            latlongs = fixed
        latlongs = np.array(latlongs)
        lats, longs = latlongs[:, 0:1], latlongs[:, 1:2]
        x = np.cos(lats) * np.cos(longs)
        y = np.cos(lats) * np.sin(longs)
        z = np.sin(lats)
        lats /= 90
        longs /= 180
        latlongs = np.hstack((lats, longs, x, y, z))
        return torch.Tensor(latlongs)


class LatLongQuantileOrdinalEnc(EncBase):
    def __init__(self, disc_params=None):
        if disc_params is not None:
            self.cat_cards = []
            self.discs = self.get_new_base_enc()
            for disc, (n_bins_, bin_edges_) in zip(self.discs, disc_params):
                disc.disc.n_bins_ = n_bins_
                disc.bin_edges_ = bin_edges_
                self.cat_cards.append(n_bins_ + 2)

    def fit(self, data):
        data = LatLongScalarEnc().enc_cont(data)
        self.cat_cards = []
        self.discs = self.get_new_base_enc()
        for col, disc in enumerate(self.discs):
            fit_data = data[:, col].numpy().reshape(-1, 1)
            disc.fit(fit_data)
            self.cat_cards.append(int(disc.n_bins_ + 2))

    def enc_cat(self, data):
        # todo: add support for missing values, which should get encoded as 1.
        data = LatLongScalarEnc().enc_cont(data)
        feats = []
        for col, disc in enumerate(self.discs):
            d = data[:, col].reshape(-1, 1)
            d = disc.transform(d).reshape(-1)
            d = d + 2  # for missing and padding
            feats.append(d)
        feats = np.stack(feats, axis=1)
        return torch.LongTensor(feats)

    def enc_cont(self, data):
        pass

    @staticmethod
    def get_new_base_enc():
        return [KBinsDiscretizer(n_bins=8,
                                 encode='ordinal',
                                 strategy='quantile') for _ in range(LatLongScalarEnc.cont_dim)]

    def get_base_enc_params(self):
        return [(disc.n_bins_, disc.bin_edges_) for disc in self.discs]


class TfidfEnc(EncBase):
    def __init__(self, vocabulary_=None, idf_=None):
        if vocabulary_ is not None and idf_ is not None:
            self.tfidf = self.get_new_base_enc()
            self.tfidf.vocabulary_ = vocabulary_
            self.tfidf.idf_ = np.array(idf_)
            self.cont_dim = len(vocabulary_)

    def enc_cat(self, data):
        pass

    def enc_cont(self, data):
        data = self.clean_data(data)
        text_strings = np.array([d if d is not None else '' for d in data])
        encoded = self.tfidf.transform(text_strings)
        encoded = torch.Tensor(encoded.todense())
        # todo: wait until pytorch lets you use multiproc with sparse tensors
        # encoded = encoded.tocoo()
        # i = torch.LongTensor(np.vstack((encoded.row, encoded.col)))
        # v = torch.FloatTensor(encoded.data)
        # encoded = torch.sparse.FloatTensor(i, v, torch.Size(encoded.shape))
        return encoded

    def fit(self, data):
        data = super().fit(data)
        data = [d if d is not None else '' for d in data]
        self.tfidf = self.get_new_base_enc().fit(data)
        self.cont_dim = len(self.tfidf.vocabulary_)

    @staticmethod
    def get_new_base_enc():
        return TfidfVectorizer(input='content',
                               decode_error='replace',
                               strip_accents='ascii',
                               lowercase=True,
                               analyzer='word',
                               min_df=5 / 100000)

    def get_base_enc_params(self):
        return self.tfidf.vocabulary_, self.tfidf.idf_


class TextSummaryScalarEnc(EncBase):
    """
    Returns the featuretools summary statistics about the text (num words and num_chars), but normalized
    """
    cont_dim = 2

    def __init__(self, center_=None, scale_=None):
        if center_ is not None and scale_ is not None:
            self.scaler = RobustScaler()
            self.scaler.center_ = center_
            self.scaler.scale_ = scale_

    def enc_cat(self, data):
        pass

    def enc_cont(self, data):
        data = self.clean_data(data)
        text_strings = [s if s is not None else '' for s in data]
        encoded = self.get_encoded(text_strings)
        encoded = self.scaler.transform(encoded)
        encoded = torch.Tensor(encoded)
        return encoded

    def get_encoded(self, text_strings):
        text_strings = [ts if ts is not None else '' for ts in text_strings]
        num_chars = [len(ts) for ts in text_strings]
        num_words = [len(ts.split()) for ts in text_strings]
        return np.array((num_chars, num_words)).T

    def fit(self, data):
        data = super().fit(data)
        encoded = self.get_encoded(data)
        self.scaler = RobustScaler().fit(encoded)

    def get_base_enc_params(self):
        return self.scaler.center_, self.scaler.scale_


class EmbeddingInitializer(nn.Module):
    def __init__(self, num_embeddings, max_emb_dim, p_dropout, minimize_emb_dim=True, drop_whole_embeddings=False,
                 one_hot=False, out_dim=None, shared_embedding=False, n_shared_embs=8,
                 shared_embedding_added=False):
        """
        :param minimize_emb_dim:
            Whether to set embedding_dim = max_emb_dim or to make embedding_dim smaller is num_embeddings is small
        :param drop_whole_embeddings:
            If True, dropout pretends the embedding was a missing value. If false, dropout sets embed features to 0
        :param one_hot:
            If True, one-hot encode variables whose cardinality is < max_emb_dim. Also, set reqiures_grad = False
        :param out_dim:
            If None, return the embedding straight from self.embed.  If another dimension, put the embedding through a
            Linear layer to make it size (batch x out_dim).
         :param shared_embedding:
            If True, 1/(n_shared_embs)th of every embedding will be reserved for a learned parameter that's common to all embeddings.
            This is useful for transformers to identify which column an embedding came from.
            Mutually exclusive with one_hot.

        Note: the 0 embedding is reserved for padding and masking.  The various encoders use 1 for missing values.

        """
        super().__init__()
        assert not (one_hot and out_dim is not None)
        self.p_dropout = p_dropout
        self.drop_whole_embeddings = drop_whole_embeddings
        self.shared_embedding = shared_embedding
        self.shared_embedding_added = shared_embedding_added
        if minimize_emb_dim or one_hot:
            self.emb_dim = min(max_emb_dim, num_embeddings)  # Don't use a crazy huge embedding if not needed
        else:
            self.emb_dim = max_emb_dim
        self.reshape_out = nn.Identity()
        if out_dim is not None:
            assert self.emb_dim <= out_dim, 'Makes no sense: just set max_emb_dim = out_dim and out_dim = None'
            if num_embeddings > self.emb_dim:
                self.reshape_out = nn.Linear(self.emb_dim, out_dim, bias=True)
            else:
                self.emb_dim = out_dim
        # Note: if you change the name of self.embed, or initialize an embedding elsewhere in a model,
        # the function get_optim will not work properly
        self.embed = nn.Embedding(num_embeddings=num_embeddings + 1,
                                  embedding_dim=self.emb_dim,
                                  padding_idx=0)
        self.embed.weight.data.clamp_(-2, 2)  # Use truncated normal init
        if one_hot:
            self.embed.weight.requires_grad = False
            if num_embeddings <= max_emb_dim:
                self.embed.weight.data[1:, :] = torch.eye(self.emb_dim)
        if shared_embedding:
            assert not one_hot
            ce_dim = self.emb_dim if shared_embedding_added else (
                                                                     out_dim if out_dim else self.emb_dim) // n_shared_embs  # used to be //8
            self.shared_emb = nn.Parameter(torch.empty(1, ce_dim).uniform_(-1, 1))
        self.do = nn.Dropout(p=p_dropout)

    def forward(self, input):
        if self.drop_whole_embeddings and self.training:
            mask = torch.zeros_like(input).bernoulli_(1 - self.p_dropout)
            input = input * mask
        out = self.embed(input)
        if not self.drop_whole_embeddings:
            out = self.do(out)
        out = self.reshape_out(out)
        if self.shared_embedding:
            shared_emb = self.shared_emb.expand(out.shape[0], -1)
            if not self.shared_embedding_added:
                out[:, :shared_emb.shape[1]] = shared_emb
            else:
                out += shared_emb
        return out


def one_hot(x, card):
    assert isinstance(x, torch.LongTensor)
    assert x.dim() == 2
    x_one_hot = x.new_zeros(x.size()[0], card).scatter_(1, x, 1)
    return x_one_hot


"""
These functions stolen wholesale, with much gratitude, from
    https://github.com/fastai/fastai/blob/master/fastai/tabular/transform.py
"""


def make_date(df: DataFrame, date_field: str):
    "Make sure `df[field_name]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)


def add_datepart(df: DataFrame, field_name: str, prefix: str = None, drop: bool = True, time: bool = False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    prefix = re.sub('[Dd]ate$', '', field_name) if prefix is None else prefix
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower())
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df


def cyclic_dt_feat_names(time: bool = True, add_linear: bool = False) -> List[str]:
    "Return feature names of date/time cycles as produced by `cyclic_dt_features`."
    fs = ['cos', 'sin']
    attr = [f'{r}_{f}' for r in 'weekday day_month month_year day_year'.split() for f in fs]
    if time: attr += [f'{r}_{f}' for r in 'hour clock min sec'.split() for f in fs]
    if add_linear: attr.append('year_lin')
    return attr


def cyclic_dt_features(d: Union[date, datetime], time: bool = True, add_linear: bool = False) -> List[float]:
    "Calculate the cos and sin of date/time cycles."
    tt, fs = d.timetuple(), [np.cos, np.sin]
    day_year, days_month = tt.tm_yday, calendar.monthrange(d.year, d.month)[1]
    days_year = 366 if calendar.isleap(d.year) else 365
    rs = d.weekday() / 7, (d.day - 1) / days_month, (d.month - 1) / 12, (day_year - 1) / days_year
    feats = [f(r * 2 * np.pi) for r in rs for f in fs]
    if time and isinstance(d, datetime) and type(d) != date:
        rs = tt.tm_hour / 24, tt.tm_hour % 12 / 12, tt.tm_min / 60, tt.tm_sec / 60
        feats += [f(r * 2 * np.pi) for r in rs for f in fs]
    if add_linear:
        if type(d) == date:
            feats.append(d.year + rs[-1])
        else:
            secs_in_year = (datetime(d.year + 1, 1, 1) - datetime(d.year, 1, 1)).total_seconds()
            feats.append(d.year + ((d - datetime(d.year, 1, 1)).total_seconds() / secs_in_year))
    return feats


def add_cyclic_datepart(df: DataFrame, field_name: str, prefix: str = None, drop: bool = True, time: bool = False,
                        add_linear: bool = False):
    "Helper function that adds trigonometric date/time features to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    prefix = re.sub('[Dd]ate$', '', field_name) if prefix is None else prefix
    series = field.apply(partial(cyclic_dt_features, time=time, add_linear=add_linear))
    columns = [prefix + c for c in cyclic_dt_feat_names(time, add_linear)]
    df_feats = pd.DataFrame([item for item in series], columns=columns)
    df = pd.concat([df, df_feats], axis=1)
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df
