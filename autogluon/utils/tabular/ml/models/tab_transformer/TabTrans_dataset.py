#Code in this script based on code from Milan Cvitkovic, 
#Xin Huang, Ashish Khetan and Zohar Karnin.

import os
from data_code.utils import get_ds_info
from data_code import data_encoders
from torchtext import data
from torch.utils.data import Dataset, DataLoader
from data_code.data_encoders import WontEncodeError, NullEnc
import torch
import pandas as pd
import numpy as np

class TabTransDataset(Dataset):
	def __init__(self, data_idx=None, random_state=0, **kwargs):
		self.ds_name  = kwargs['dataset_name']
		self.encoders = kwargs['encoders']
		self.kwargs   = kwargs
		self.data_idx = data_idx
		self.ds_info  = get_ds_info(self.ds_name)

		raw_data_path = self.ds_info['processed']['local_path'] 

		dataset_size = kwargs['dataset_size']
		short_data_path = raw_data_path + str(dataset_size)
		if self.ds_info['processed']['format']==".csv": 
			df = pd.read_csv(raw_data_path)		
		if self.ds_info['processed']['format']==".parquet": 
			df = pd.read_parquet(raw_data_path)

		if self.ds_name=='uomi-unlabeled':
			del df['item_id']
			del df['marketplace_id']
			del df['list_price_currency']
			df = df.rename(columns={'item_name': 'product_name'})

		#del df['department']
		#del df['gl_product_group_type']
		#del df['item_type_keyword']
		
		self.raw_data = df.iloc[:dataset_size]
		self.raw_data     = self.raw_data.iloc[self.data_idx]
		print("Dataset size: {}".format(len(self.raw_data)))
		self.columns = self.ds_info['meta']['columns'][1:]

		def tfm(x):
			if x in ['0', '-1', '0.0', 'no', 'No', 'neg', 'n', 'N', 'False', 'NRB', ' <=50K']:
				return 0
			elif x == 'nan':
				return np.nan
			for i in range(1,self.ds_info['processed']['classes']):
				if x in [str(i), str(i)+'.0']:
					return int(i)
			else:
				return 1

		self.targets = None
	
		try: #will fail if dataset is unlabeled 
			targets = self.raw_data['TARGET'].loc[self.data_idx].astype(int).astype(str).transform(tfm).values   
			self.targets = torch.LongTensor(targets)
		except:
			pass

		self.cat_feat_origin_cards = None
		self.cont_feat_origin = None
		self.feature_encoders = None

		self.fit_feat_encoders()
		self.encode(self.feature_encoders)


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
			print(feature_encoders.keys())

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

	def build_loader(self):
		loader = DataLoader(self, batch_size=self.kwargs['batch_size'], 
							shuffle=False, num_workers=16, 
							pin_memory=True) 
		return loader

	def __len__(self):
		return len(self.data_idx)

	def __getitem__(self, idx):    
		target = self.targets[idx] if self.targets is not None else []
		input   = self.cat_data[idx]  if self.cat_data is not None else []
		return input, target

	def data(self):
		return self.raw_data

