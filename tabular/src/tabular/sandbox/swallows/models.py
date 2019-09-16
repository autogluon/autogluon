from fastai.tabular import *
from fastai.text import *
from swallows.models import *
from swallows.training import *


# ================================================================= New stuff


def load_nlp_backbone(data_text, encoder):
    learn = text_classifier_learner(data_text, AWD_LSTM, drop_mult=0.5)
    learn.load_encoder(encoder)
    model = learn.model
    model[-1].layers = model[-1].layers[:-3]
    return model, learn.layer_groups


def load_nlp_backbone_pretrained(data_text, pretrained_data, drop_mult=0.5):
    learn = text_classifier_learner(data_text, AWD_LSTM, drop_mult=drop_mult)
    learn.load(pretrained_data)
    model = learn.model
    model[-1].layers = model[-1].layers[:-3]
    return model, learn.layer_groups


def load_tabular_backbone(data_tab, tab_model):
    learn = tabular_learner(data_tab, layers=[727, 353], ps=0.2, emb_drop=0.2, metrics=accuracy)
    learn.load(tab_model)
    model = learn.model
    model.layers = model.layers[:-3]
    return model, learn.layer_groups


def get_data_features(df):
    cont_names = []
    exclude_from_cats = [
                            'ID',
                            'assigned_date',
                            'create_date',
                            'details',
                            'short_description',
                            'amazon_domains',
                            'root_cause',
                            'create_to_assigned_lag'
                            'create_to_assigned_lag_log',
                            'assigned_date_Elapsed'
                            'create_date_Elapsed',
                        ] + cont_names

    cat_names = [c for c in df.columns if c not in exclude_from_cats]
    dep_var = 'root_cause'
    procs = [FillMissing, Categorify, Normalize]
    return cont_names, cat_names, dep_var, procs


def get_data_features_v2(df):
    cont_names = [col for col in df.columns if '_ratio' in col]
    exclude_from_cats = [
                            'ID',
                            'assigned_date',
                            'create_date',
                            'details',
                            'short_description',
                            'amazon_domains',
                            'root_cause',
                            'create_to_assigned_lag'
                            'create_to_assigned_lag_log',
                            'assigned_date_Elapsed'
                            'create_date_Elapsed',
                        ] + cont_names

    cat_names = [c for c in df.columns if c not in exclude_from_cats]
    dep_var = 'root_cause'
    procs = [FillMissing, Categorify, Normalize]
    return cont_names, cat_names, dep_var, procs


def accuracy_v2(input: Tensor, targs: Tensor) -> Rank0Tensor:
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n, -1)
    targs = targs.view(n, -1)
    return (input == targs).float().mean()


class ModifiedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(ModifiedCrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        import ipdb;
        ipdb.set_trace()
        return F.cross_entropy(input[0], target)
