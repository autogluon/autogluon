import os
import random
from math import ceil
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

#from __init__ import data_root
from data_code import data_encoders
from data_code.data_encoders import WontEncodeError, NullEnc
from data_code.utils import get_ds_info


class TabularDataset(Dataset):
    def __init__(self, dataset_name=None,  encoders=None):
        self.ds_name = dataset_name
        self.encoders = encoders
        self.ds_info = get_ds_info(dataset_name)

        raw_data_path = self.ds_info['processed']['local_path'] #os.path.join(data_root, self.ds_info['processed']['local_path'])


        col_names = [c['name'] for c in self.ds_info['meta']['columns']]
        self.raw_data = pd.read_csv(raw_data_path) 

        self.tabular_data = None
        self.text_data    = None
        self.image_data   = None 

        
        def tfm(x):
            if x in ['0', '-1', '0.0', 'no', 'No', 'neg', 'n', 'N', 'False', 'NRB', ' <=50K']:
                return 0
            elif x == 'nan':
                return pd.np.nan
            elif x == 'unlabel':
                return -1
            for i in range(1,self.ds_info['processed']['classes']):
                if x == str(i):
                    return i 
            else:
                return 1

        targets = self.raw_data['TARGET'].astype(str).transform(tfm)
       
        targets = np.array(targets).astype(np.float)
        self.targets = torch.LongTensor(targets)


        self.raw_data = self.raw_data[[i for i in self.raw_data.columns if i != 'TARGET']]


        self.columns = self.ds_info['meta']['columns'][1:]  # Omitting the target column
        self.cat_feat_origin_cards = None
        self.cont_feat_origin = None
        self.feature_encoders = None
        self.locs_to_replace = None
        self.random_cat_feats = None


    @property
    def n_cont_features(self):
        return len(self.cont_feat_origin) if self.encoders is not None else None

    def fit_feat_encoders(self):
        if self.encoders is not None:
            self.feature_encoders = {}
            for c in self.columns:
                col = self.raw_data[c['name']]
                enc = data_encoders.__dict__[self.encoders[c['type']]]()
                if c['type'] == 'SCALAR' and col.nunique() < 32:
                    print(f"Column {c['name']} shouldn't be encoded as SCALAR. Switching to CATEGORICAL.")
                    enc = data_encoders.__dict__[self.encoders['CATEGORICAL']]()
                try:
                    enc.fit(col)
                except WontEncodeError as e:
                    print(f"Not encoding column '{c['name']}': {e}")
                    enc = NullEnc()
                self.feature_encoders[c['name']] = enc

    def encode(self, feature_encoders):
        if self.encoders is not None:
            self.feature_encoders = feature_encoders
            self.cat_feat_origin_cards = []
            cat_features = []
            self.cont_feat_origin = []
            cont_features = []
            for c in self.columns:
                enc = feature_encoders[c['name']]
                col = self.raw_data[c['name']]
                cat_feats = enc.enc_cat(col)
                if cat_feats is not None:
                    self.cat_feat_origin_cards += [(f'{c["name"]}_{i}_{c["type"]}', card) for i, card in
                                                   enumerate(enc.cat_cards)]
                    cat_features.append(cat_feats)
                cont_feats = enc.enc_cont(col)
                if cont_feats is not None:
                    self.cont_feat_origin += [c['name']] * enc.cont_dim
                    cont_features.append(cont_feats)
            if cat_features:
                self.cat_data = torch.cat(cat_features, dim=1)
            else:
                self.cat_data = None
            if cont_features:
                self.cont_data = torch.cat(cont_features, dim=1)
            else:
                self.cont_data = None



    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item: int):
        target = self.targets[item]
        if self.encoders is not None:
            cat_feats = self.cat_data[item, :] if self.cat_data is not None else []
            cont_feats = self.cont_data[item, :] if self.cont_data is not None else []

            if self.locs_to_replace is None:
                input = cat_feats, cont_feats
            else:
                input = cat_feats, cont_feats, self.locs_to_replace[item, :], self.random_cat_feats[item, :]
            return input, target
        else:
            return ' '.join(str(i) for i in self.raw_data.iloc[item, :].to_list()), target

