import torch
from itertools import product
import torch.nn as nn
from copy import deepcopy
import numpy as np

class CONTRASTIVE_pretext(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

    def pairwise_loss(out, temperature):
        #TODO: implement for arbitrary modalities
        out_1, out_2 = out[:,0,:], out[:,1,:]
        len_out = out_1.shape[0]
        out = out_1, out_2 
        out = torch.cat(out, dim=0)
        
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        pos_sim = torch.cat([sim_matrix[:len_out,len_out:].diag(),sim_matrix[len_out:,:len_out].diag()])
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss

    def forward(self, out, target):
        return self.pairwise_loss(out, self.kwargs['temperature']), None

    def get(self, data, target):
        data  = data.to(device,self.kwargs['device'], non_blocking=True) 
        return data, None


class SUPERVISED_pretext(nn.Module):
    #holder class to handle supervised learning in the framework of "pretext" tasks.
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs    = kwargs
        self.cross_ent = nn.CrossEntropyLoss()


    def forward(self, out, target):
        loss = self.cross_ent(out,target)
        pred    = out.max(dim=1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum()
        correct = correct.float()
        correct = correct/out.shape[0]
        return loss, correct

    def get(self, data, target):
        data   = data.to(self.kwargs['device'], non_blocking=True) 
        target = target.to(self.kwargs['device'], non_blocking=True)
        return data, target


class BERT_pretext(nn.Module):
    def __init__(self, kwargs, replacement_noise='random', p_replace=0.3):
        super().__init__()
        self.kwargs            = kwargs
        self.cross_ent         = nn.CrossEntropyLoss()
        self.p_replace         = p_replace
        self.predicters        = nn.ModuleList()
        self.n_cat_feats       = len(kwargs['cat_feat_origin_cards'])
        self.replacement_noise = replacement_noise

        number_devices = torch.cuda.device_count()
        for col in range(self.n_cat_feats):
            lin = nn.Linear(kwargs['tab_kwargs']['hidden_dim'], 2)
            self.predicters.append(lin)

    def forward(self, out, target):
        prob = torch.cat([self.predicters[col](out[:,col,:]).unsqueeze(1) for col in range(self.n_cat_feats)],dim=1)
        prob   = prob.view(-1,2)
        target = target.view(-1)

        loss    = self.cross_ent(prob,target)
        pred    = prob.max(dim=1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum()
        correct = correct.float()
        correct = correct/pred.shape[0]

        return loss, correct

    def get(self, data, target):
        cat_feats = data

        orig_cat_feats = deepcopy(cat_feats.detach())

        if self.replacement_noise=='swap':
            n_cat = cat_feats.shape[1]
            cols_to_shuffle = np.random.choice(n_cat, int(self.p_replace * n_cat), replace=False)
            for col in cols_to_shuffle:
                cat_feats[:, col] = cat_feats[:, col][torch.randperm(cat_feats.shape[0])]

        elif self.replacement_noise=='random':
            locs_to_replace = torch.empty_like(cat_feats, dtype=float).uniform_() < self.p_replace
            col_cardinalities = torch.LongTensor([i[1] for i in self.kwargs['cat_feat_origin_cards']]).to(cat_feats)
            col_cardinalities = col_cardinalities.unsqueeze(0).expand_as(cat_feats)

            unif = torch.rand(cat_feats.shape, device=col_cardinalities.device)
            random_feats = (unif * col_cardinalities).floor().to(torch.int64) + 1  # + 1 since 0 is the padding value

            extra_replace = torch.mul((cat_feats == random_feats).to(int), locs_to_replace.to(int)).to(torch.bool)
            cat_feats[locs_to_replace] = random_feats[locs_to_replace]

            assert torch.all(cat_feats[extra_replace] == orig_cat_feats[extra_replace]).item() is True
            extra_plus1 = cat_feats[extra_replace] + 1
            extra_minus1 = cat_feats[extra_replace] - 1
            extra_zero_padd_idx = extra_minus1 == 0
            extra_minus1[extra_zero_padd_idx] = extra_plus1[extra_zero_padd_idx]

            cat_feats[extra_replace] = extra_minus1
            assert torch.all(~(cat_feats[extra_replace] == orig_cat_feats[extra_replace])).item() is True

        elif self.replacement_noise=='low_rank':
                    assert self.p_replace + 0.2 <= 1, "p_replace too big, lower it!"
                    weights = torch.tensor([self.p_replace, 0.1, 0.9-self.p_replace], dtype=torch.float) #0=pad, 1=replace with random value, 2=dont change

                    locs_to_change = torch.multinomial(weights, np.prod(cat_feats.shape),replacement=True).view(cat_feats.shape)
                    col_cardinalities = torch.LongTensor([i[1] for i in self.kwargs['cat_feat_origin_cards']]).to(cat_feats)
                    col_cardinalities = col_cardinalities.unsqueeze(0).expand_as(cat_feats)

                    unif = torch.rand(cat_feats.shape, device=col_cardinalities.device)
                    random_feats = (unif * col_cardinalities).floor().to(torch.int64) + 1  # + 1 since 0 is the padding value

                    extra_replace = torch.mul((cat_feats == random_feats).to(int), (locs_to_change==1).to(int)).to(torch.bool)
                    cat_feats[locs_to_change==1] = random_feats[locs_to_change==1]
                    cat_feats[locs_to_change==0] = 0 

                    assert torch.all(cat_feats[extra_replace] == orig_cat_feats[extra_replace]).item() is True
                    extra_plus1 = cat_feats[extra_replace] + 1
                    extra_minus1 = cat_feats[extra_replace] - 1
                    extra_zero_padd_idx = extra_minus1 == 0
                    extra_minus1[extra_zero_padd_idx] = extra_plus1[extra_zero_padd_idx]

                    cat_feats[extra_replace] = extra_minus1
                    assert torch.all(~(cat_feats[extra_replace] == orig_cat_feats[extra_replace])).item() is True

        pretext_label = (cat_feats != orig_cat_feats).long()
        pretext_data = cat_feats

        pretext_data  = pretext_data.to(self.kwargs['device'], non_blocking=True) 
        pretext_label = pretext_label.to(self.kwargs['device'], non_blocking=True)

        return pretext_data, pretext_label


