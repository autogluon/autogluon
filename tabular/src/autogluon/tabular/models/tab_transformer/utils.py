import torch
from torch.utils.data import Dataset, DataLoader

from autogluon.core.utils.exceptions import TimeLimitExceeded
from . import tab_transformer_encoder
from .tab_transformer_encoder import WontEncodeError, NullEnc
import logging
from autogluon.core import args
from ..abstract import model_trial

logger = logging.getLogger(__name__)

@args()
def tt_trial(args, reporter):
    """ Training and evaluation function used during a single trial of HPO """
    try:
        model, args, util_args = model_trial.prepare_inputs(args=args)

        fit_model_args = dict(X_train=util_args.X_train, y_train=util_args.y_train, X_val=util_args.X_val, y_val=util_args.y_val)
        predict_proba_args = dict(X=util_args.X_val)
        model = model_trial.fit_and_save_model(model=model, params=args, fit_args=fit_model_args, predict_proba_args=predict_proba_args, y_val=util_args.y_val,
                                       time_start=util_args.time_start, time_limit=util_args.get('time_limit', None), reporter=reporter)
    except Exception as e:
        if not isinstance(e, TimeLimitExceeded):
            logger.exception(e, exc_info=True)
        reporter.terminate()
    else:
        reporter(epoch = model.params['epochs'] + 1, validation_performance=model.val_score)
    pass

def augmentation(data, target, **params):
    shape = data.shape
    cat_data = torch.cat([data for _ in range(params['num_augs'])])
    target = torch.cat([target for _ in range(params['num_augs'])]).view(-1)
    locs_to_mask = torch.empty_like(cat_data, dtype=float).uniform_() < params['aug_mask_prob']
    cat_data[locs_to_mask] = 0
    cat_data = cat_data.view(-1, shape[-1])
    return cat_data, target


def get_col_info(X):
    """
    If we somehow do not get col_info when creating a TT dataset, then set every feature type to CATEGORICAL.
    """
    cols = list(X.columns)
    col_info = []
    for c in cols:
        col_info.append({"name": c, "type": "CATEGORICAL"})
    return col_info


class TabTransformerDataset(Dataset):
    def __init__(self, X, encoders, problem_type, y=None, col_info=None):
        self.encoders = encoders
        self.col_info = col_info

        self.raw_data = X

        if y is None:
            self.targets = None
        elif problem_type == REGRESSION:
            self.targets = torch.FloatTensor(y)
        else:
            self.targets = torch.LongTensor(y)

        if col_info is None:
            self.columns = get_col_info(X)
        else:
            self.columns = self.col_info

        """must be a list of dicts, each dict is of the form {"name": col_name, "type": col_type} 
        where col_name is obtained from the df X, and col_type is CATEGORICAL, TEXT or SCALAR
         """
        self.cat_feat_origin_cards = None
        self.cont_feat_origin = None
        self.feature_encoders = None

    @property
    def n_cont_features(self):
        return len(self.cont_feat_origin) if self.encoders is not None else None

    def fit_feat_encoders(self):
        if self.encoders is not None:
            self.feature_encoders = {}
            for c in self.columns:
                col = self.raw_data[c['name']]
                enc = tab_transformer_encoder.__dict__[self.encoders[c['type']]]()

                if c['type'] == 'SCALAR' and col.nunique() < 32:
                    logger.log(15, f"Column {c['name']} shouldn't be encoded as SCALAR. Switching to CATEGORICAL.")
                    enc = tab_transformer_encoder.__dict__[self.encoders['CATEGORICAL']]()
                try:
                    enc.fit(col)
                except WontEncodeError as e:
                    logger.log(15, f"Not encoding column '{c['name']}': {e}")
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

    def build_loader(self, batch_size, num_workers, shuffle=False):
        loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        loader.cat_feat_origin_cards = self.cat_feat_origin_cards
        return loader

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        target = self.targets[idx] if self.targets is not None else []
        input = self.cat_data[idx] if self.cat_data is not None else []
        return input, target

    def data(self):
        return self.raw_data
