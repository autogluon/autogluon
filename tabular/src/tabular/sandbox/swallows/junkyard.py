from fastai.text import *
from swallows.models import *
from swallows.training import *
from torch.utils.data import SequentialSampler


# These are more generic, but 50% slower (loop takes 22 minutes vs 14 mins to train one cycle)

class CompositeModel(nn.Module):
    def __init__(self, tabular_models, nlp_models, layers, drops):
        super().__init__()
        self.tabular_models = tabular_models
        self.nlp_models = nlp_models
        lst_layers = []
        activs = [nn.ReLU(inplace=True), ] * (len(layers) - 2) + [None]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            lst_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*lst_layers)

    def forward(self, *x):
        xs_tabular = [model(*x[i]) for i, model in enumerate(self.tabular_models)]
        xs_nlp = [model(x[i + len(self.tabular_models)])[0] for i, model in enumerate(self.nlp_models)]
        x = torch.cat(xs_tabular + xs_nlp, dim=1)
        x = self.layers(x)
        return x


def select_sampler(ds, ds_xs, sources, sampler_type, sampler_source_id, bs):
    if sampler_type == 'sortish' or sampler_type == 'sort':
        if sampler_source_id < 0:
            raise ValueError('sampler_source_id should be specified for non-sequential sampler')
        if sampler_type == 'sortish':
            sampler = SortishSampler(ds_xs[sampler_source_id], key=lambda t: len(sources[sampler_source_id].valid_ds[t][0].data), bs=bs // 2)
        elif sampler_type == 'sort':
            sampler = SortSampler(ds_xs[sampler_source_id], key=lambda t: len(sources[sampler_source_id].valid_ds[t][0].data), bs=bs // 2)
    else:
        sampler = SequentialSampler(ds)
    return sampler


def composite_data_bunch(path, sources, collate_fn,
                         train_sampler_type='sequential', train_sampler_source_id=-1,
                         valid_sampler_type='sequential', valid_sampler_source_id=-1,
                         bs=64):
    # !!!! TABULAR SOURCES MUST GO FIRST !!!!

    train_ds_xs = [source.train_ds.x for source in sources]
    valid_ds_xs = [source.valid_ds.x for source in sources]
    classes = sources[0].classes

    train_ds = CompositeDataset(train_ds_xs, sources[0].train_ds.y, classes)
    valid_ds = CompositeDataset(valid_ds_xs, sources[0].valid_ds.y, classes)

    train_sampler = select_sampler(train_ds, train_ds_xs, sources, train_sampler_type, train_sampler_source_id, bs)
    valid_sampler = select_sampler(valid_ds, valid_ds_xs, sources, valid_sampler_type, valid_sampler_source_id, bs)

    train_dl = DataLoader(train_ds, bs // 2, sampler=train_sampler)
    valid_dl = DataLoader(valid_ds, bs, sampler=valid_sampler)

    test_dl = None
    if sources[0].test_ds is not None:
        test_ds_xs = [source.test_ds.x for source in sources]
        test_ds = CompositeDataset(test_ds_xs, sources[0].test_ds.y, classes)
        test_dl = DataLoader(test_ds, bs, sampler=(SequentialSampler(test_ds)))

    return DataBunch(train_dl, valid_dl, test_dl=test_dl, device=defaults.device, collate_fn=collate_fn, path=path)


class CompositeDataset(Dataset):
    def __init__(self, xs, y, classes):
        self.xs = xs
        self.y = y
        self.classes = classes
        self.c = len(classes)

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return tuple([x[i] for x in self.xs]), self.y[i]


# ============================================== DEPRECATED STUFF

# TODO: DEPRECATED
def collate_ensemble_v2(batch):
    x, y = list(zip(*batch))
    x1, x2 = list(zip(*x))

    # nlp source
    x1, y = pad_collate(list(zip(x1, y)), pad_idx=1, pad_first=True)

    # tabular source
    x2 = to_data(x2)
    x2 = list(zip(*x2))
    x2 = torch.stack(x2[0]), torch.stack(x2[1])

    return (x1, x2), y


# TODO: DEPRECATED
def collate_ensemble_v3(batch):
    x, y = list(zip(*batch))
    x1, x2, x3 = list(zip(*x))

    # nlp source - details
    x1, y = pad_collate(list(zip(x1, y)), pad_idx=1, pad_first=True)

    # tabular source
    x2 = to_data(x2)
    x2 = list(zip(*x2))
    x2 = torch.stack(x2[0]), torch.stack(x2[1])

    # nlp source - short_description
    x3, y = pad_collate(list(zip(x3, y)), pad_idx=1, pad_first=True)

    return (x1, x2, x3), y


# TODO: DEPRECATED
class EnsembleModel(nn.Module):
    def __init__(self, mod_tab, mod_nlp, layers, drops):
        super().__init__()
        self.mod_tab = mod_tab
        self.mod_nlp = mod_nlp
        lst_layers = []
        activs = [nn.ReLU(inplace=True), ] * (len(layers) - 2) + [None]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            lst_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*lst_layers)

    def forward(self, *x):
        x_nlp = self.mod_nlp(x[0])[0]
        x_tab = self.mod_tab(*x[1])
        x = torch.cat([x_nlp, x_tab], dim=1)
        x = self.layers(x)
        return x


# TODO: DEPRECATED
class EnsembleModel3(nn.Module):
    def __init__(self, tab_model, nlp_details_model, nlp_descr_model, layers, drops):
        super().__init__()
        self.tab_model = tab_model
        self.nlp_details_model = nlp_details_model
        self.nlp_descr_model = nlp_descr_model
        lst_layers = []
        activs = [nn.ReLU(inplace=True), ] * (len(layers) - 2) + [None]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            lst_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*lst_layers)

    def forward(self, *x):
        # Inputs: details, tabular, short_description
        x_nlp_details = x[0]
        x_tab = x[1]
        x_nlp_short_description = x[2]

        x_nlp_details = self.nlp_details_model(x_nlp_details)[0]
        x_tab = self.tab_model(*x_tab)
        x_nlp_short_description = self.nlp_descr_model(x_nlp_short_description)[0]
        x = torch.cat([x_nlp_details, x_tab, x_nlp_short_description], dim=1)
        x = self.layers(x)
        return x


# TODO: DEPRECATED
class EnsembleDataset(Dataset):
    def __init__(self, x_text, x_tab, y, classes):
        self.x_text = x_text
        self.x_tab = x_tab
        self.y = y
        self.classes = classes
        self.c = len(classes)

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return (self.x_text[i], self.x_tab[i]), self.y[i]


# TODO: DEPRECATED
class EnsembleDataset3(Dataset):
    def __init__(self, x_text_details, x_tab, x_text_descr, y, classes):
        self.x_text_details = x_text_details
        self.x_text_descr = x_text_descr
        self.x_tab = x_tab
        self.y = y
        self.classes = classes
        self.c = len(classes)

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return (self.x_text_details[i], self.x_tab[i], self.x_text_descr[i]), self.y[i]


# TODO: DEPRECATED
def mixed_data_bunch(path, data_text, data_tab, bs):
    train_ds = EnsembleDataset(data_text.train_ds.x, data_tab.train_ds.x, data_tab.train_ds.y, data_text.classes)
    valid_ds = EnsembleDataset(data_text.valid_ds.x, data_tab.valid_ds.x, data_tab.valid_ds.y, data_text.classes)

    train_sampler = SortishSampler(data_text.train_ds.x, key=lambda t: len(data_text.train_ds[t][0].data), bs=bs // 2)
    valid_sampler = SequentialSampler(data_text.valid_ds.x, key=lambda t: len(data_text.valid_ds[t][0].data))

    train_dl = DataLoader(train_ds, bs // 2, sampler=train_sampler)
    valid_dl = DataLoader(valid_ds, bs, sampler=valid_sampler)

    test_dl = None
    if data_text.test_ds is not None:
        test_ds = EnsembleDataset(data_text.test_ds.x, data_tab.test_ds.x, data_tab.test_ds.y, data_text.classes)
        test_sampler = SequentialSampler(test_ds)
        test_dl = DataLoader(test_ds, bs, sampler=test_sampler)

    data = DataBunch(train_dl, valid_dl, test_dl=test_dl, device=defaults.device, collate_fn=collate_ensemble_v2, path=path)

    return data


# TODO: DEPRECATED
def mixed_data_bunch3(path, data_text_details, data_tab, data_text_descr, bs):
    train_ds = EnsembleDataset3(data_text_details.train_ds.x, data_tab.train_ds.x, data_text_descr.train_ds.x, data_tab.train_ds.y, data_text_details.classes)
    valid_ds = EnsembleDataset3(data_text_details.valid_ds.x, data_tab.valid_ds.x, data_text_descr.valid_ds.x, data_tab.valid_ds.y, data_text_details.classes)

    train_sampler = SortishSampler(data_text_details.train_ds.x, key=lambda t: len(data_text_details.train_ds[t][0].data), bs=bs // 2)
    valid_sampler = SequentialSampler(data_text_details.valid_ds.x, key=lambda t: len(data_text_details.valid_ds[t][0].data))

    train_dl = DataLoader(train_ds, bs // 2, sampler=train_sampler)
    valid_dl = DataLoader(valid_ds, bs, sampler=valid_sampler)

    test_dl = None
    if data_text_details.test_ds is not None:
        test_ds = EnsembleDataset3(data_text_details.test_ds.x, data_tab.test_ds.x, data_text_descr.test_ds.x, data_tab.test_ds.y, data_text_details.classes)
        test_sampler = SequentialSampler(test_ds)
        test_dl = DataLoader(test_ds, bs, sampler=test_sampler)

    data = DataBunch(train_dl, valid_dl, test_dl=test_dl, device=defaults.device, collate_fn=collate_ensemble_v3, path=path)

    return data

