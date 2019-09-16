from fastai.tabular import tabular_learner
from fastai.text import *
from torch.utils.data import SequentialSampler


def package_tabular_inputs(x):
    x = to_data(x)
    x = list(zip(*x))
    x = torch.stack(x[0]), torch.stack(x[1])
    return x


def package_nlp_inputs(x, y):
    return pad_collate(list(zip(x, y)), pad_idx=1, pad_first=True)


def collate_ensemble_v4(batch):
    x, y = list(zip(*batch))
    x_tab, x_short_description, x_details = list(zip(*x))

    # tabular source
    x_tab = package_tabular_inputs(x_tab)

    # nlp source - short_description
    x_short_description, _ = package_nlp_inputs(x_short_description, y)

    # nlp source - details
    x_details, y = package_nlp_inputs(x_details, y)

    return (x_tab, x_short_description, x_details), y


class EnsembleModelV4(nn.Module):
    def __init__(self, tab_model, nlp_short_description_model, nlp_details_model, layers, drops):
        super().__init__()
        self.tab_model = tab_model
        self.nlp_short_description_model = nlp_short_description_model
        self.nlp_details_model = nlp_details_model
        lst_layers = []
        activs = [nn.ReLU(inplace=True), ] * (len(layers) - 2) + [None]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            lst_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*lst_layers)

    def forward(self, *x):
        # Inputs: details, tabular, short_description
        x_tab = x[0]
        x_nlp_short_description = x[1]
        x_nlp_details = x[2]

        x_tab = self.tab_model(*x_tab)
        x_nlp_short_description = self.nlp_short_description_model(x_nlp_short_description)[0]
        x_nlp_details = self.nlp_details_model(x_nlp_details)[0]

        x = torch.cat([x_nlp_details, x_tab, x_nlp_short_description], dim=1)

        x = self.layers(x)
        return x


class EnsembleDatasetV4(Dataset):
    def __init__(self, x_tab, x_text_short_description, x_text_details, y, classes):
        self.x_tab = x_tab
        self.x_text_short_description = x_text_short_description
        self.x_text_details = x_text_details
        self.y = y
        self.classes = classes
        self.c = len(classes)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        xs = (
            self.x_tab[i],
            self.x_text_short_description[i],
            self.x_text_details[i],
        )

        return xs, self.y[i]


def mixed_data_bunch_v4(path, tab, text_short_description, text_details, bs=128):
    train_ds = EnsembleDatasetV4(
        tab.train_ds.x,
        text_short_description.train_ds.x,
        text_details.train_ds.x,
        tab.train_ds.y,
        text_details.classes)

    valid_ds = EnsembleDatasetV4(
        tab.valid_ds.x,
        text_short_description.valid_ds.x,
        text_details.valid_ds.x,
        tab.valid_ds.y,
        text_details.classes)

    train_sampler = SortishSampler(text_details.train_ds.x, key=lambda t: len(text_details.train_ds[t][0].data), bs=bs // 2)
    # valid_sampler = SortSampler(text_details.valid_ds.x, key=lambda t: len(text_details.valid_ds[t][0].data))
    valid_sampler = SequentialSampler(valid_ds)

    train_dl = DataLoader(train_ds, bs // 2, sampler=train_sampler)
    valid_dl = DataLoader(valid_ds, bs, sampler=valid_sampler)

    test_dl = None
    if text_details.test_ds is not None:
        test_ds = EnsembleDatasetV4(
            tab.test_ds.x,
            text_short_description.test_ds.x,
            text_details.test_ds.x,
            tab.test_ds.y,
            text_details.classes)
        test_sampler = SequentialSampler(test_ds)
        test_dl = DataLoader(test_ds, bs, sampler=test_sampler)

    return DataBunch(train_dl, valid_dl, test_dl=test_dl, device=defaults.device, collate_fn=collate_ensemble_v4, path=path)


def build_ensemble_v4_learner(path, models, sources, fold, bs=64):
    tab_model, tab_layer_groups = load_tabular_backbone(sources[0], f'{models[0]}_fold_{fold}_fitted')
    nlp_descr_model, nlp_dsc_layer_groups = load_nlp_backbone_pretrained(sources[1], f'{models[1]}_fold_{fold}_fitted', drop_mult=0.0, lin_ftrs=[200])
    nlp_details_model, nlp_det_layer_groups = load_nlp_backbone_pretrained(sources[2], f'{models[2]}_fold_{fold}_fitted', drop_mult=0.0, lin_ftrs=[200])

    len_nlp_det_out = nlp_details_model[1].layers[1].out_features
    len_nlp_desc_out = nlp_descr_model[1].layers[1].out_features
    len_tab_out = tab_model.layers[4].out_features

    lin_layers = [len_nlp_det_out + len_nlp_desc_out + len_tab_out, len(sources[0].classes)]
    model = EnsembleModelV4(tab_model, nlp_descr_model, nlp_details_model, layers=lin_layers, drops=[0.2])
    layer_groups = [
        nn.Sequential(*(flatten_model(nlp_det_layer_groups[0]) + flatten_model(nlp_dsc_layer_groups[0]) + flatten_model(tab_layer_groups[0]))),
        nn.Sequential(*(flatten_model(nlp_det_layer_groups[1]) + flatten_model(nlp_dsc_layer_groups[1]))),
        nn.Sequential(*(flatten_model(nlp_det_layer_groups[2]) + flatten_model(nlp_dsc_layer_groups[2]))),
        nn.Sequential(*(flatten_model(nlp_det_layer_groups[3]) + flatten_model(nlp_dsc_layer_groups[3]))),
        nn.Sequential(*(flatten_model(nlp_det_layer_groups[4]) + flatten_model(nlp_dsc_layer_groups[4]))),
        nn.Sequential(*flatten_model(model.layers))]

    data = mixed_data_bunch_v4(path, *sources, bs=bs)
    learn = Learner(data, model, metrics=accuracy_v2, layer_groups=layer_groups)
    learn.freeze()
    return learn


def train_ensemble_folds(path, folds, model_name_prefix, fold_train_func, bs, data_suffix=''):
    print('--------------------------------------------------------------------')
    print('!!!!! MAKE SURE THE MODEL IS SAVED IN fold_train_func FUNCTION !!!!!')
    print('--------------------------------------------------------------------')
    for fold in range(folds):
        print(f'---------------------------------------------------------------------------------------------------- fold {fold + 1} / {folds}')
        model_name = f'{model_name_prefix}_fold_{fold}'
        output_model_name = f'{model_name}_fitted'
        print(f'Training {model_name} -> {output_model_name}')

        models, sources, classes = load_ensemble_data(path, fold, data_suffix, bs)
        learn = build_ensemble_v4_learner(path, models, sources, fold, bs=bs)
        fold_train_func(learn, output_model_name)

        # Cleanup
        del learn
        gc.collect()
        torch.cuda.empty_cache()


def load_ensemble_data(path, fold, data_suffix, bs):
    models = ['data_tab', 'data_short_description', 'data_details']
    sources = [load_data(path, f'{model}_fold_{fold}{data_suffix}', bs=bs) for model in models]
    data = mixed_data_bunch_v4(path, *sources, bs=bs)
    print(f'Loaded training data | train: {len(data.train_ds)} | valid: {len(data.valid_ds)} | test {len(data.test_ds)}')
    classes = data.train_ds.classes
    return models, sources, classes


def predict_on_ensemble_folds(path, folds, model_name_prefix, bs, data_suffix=''):
    for fold in range(folds):
        print(f'---------------------------------------------------------------------------------------------------- fold {fold + 1} / {folds}')
        model_name = f'{model_name_prefix}_fold_{fold}'
        fitted_model_name = f'{model_name}_fitted'

        # Test+oof predictions
        models, sources, classes = load_ensemble_data(path, fold, data_suffix, bs)
        columns = [f'{i}' for i in range(len(classes))]
        learn = build_ensemble_v4_learner(path, models, sources, fold, bs=bs)
        learn.load(fitted_model_name)

        out_path = path / f'preds/test_{model_name}.parquet'
        preds, y = learn.get_preds(ds_type=DatasetType.Test)
        pd.DataFrame(preds.numpy(), columns=columns).to_parquet(out_path, engine='fastparquet')
        print(f'Recorded test predictions to {out_path}')

        preds, y = learn.get_preds(ds_type=DatasetType.Valid)
        out_path = path / f'preds/oof_{model_name}.parquet'
        pd.DataFrame(preds.numpy(), columns=columns).to_parquet(out_path, engine='fastparquet')
        print(f'Recorded OOF predictions to {out_path}')

        # Cleanup
        del learn
        gc.collect()
        torch.cuda.empty_cache()


def load_nlp_backbone(data_text, encoder, drop_mult=0.5, lin_ftrs=[50]):
    learn = text_classifier_learner(data_text, AWD_LSTM, drop_mult=drop_mult, lin_ftrs=lin_ftrs)
    learn.load_encoder(encoder)
    model = learn.model
    model[-1].layers = model[-1].layers[:-3]
    return model, learn.layer_groups


def load_nlp_backbone_pretrained(data_text, pretrained_data, drop_mult=0.5, lin_ftrs=[50]):
    learn = text_classifier_learner(data_text, AWD_LSTM, drop_mult=drop_mult, lin_ftrs=lin_ftrs)
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
        return F.cross_entropy(input[0], target)
