import torch
from .torch_utils import *

def init_default_train_args(net, epochs, base_lr, iters_per_epoch, criterion):
    train_args = {}
    train_args['lr_scheduler'] = LR_Scheduler('cos', base_lr, epochs, iters_per_epoch)
    train_args['optimizer'] = torch.optim.SGD(net.parameters(), lr=base_lr,
                                              momentum=0.9, weight_decay=1e-4)
    train_args['criterion'] = criterion
    return train_args

def default_val_fn(net, data, label, metric, use_cuda):
    net.eval()
    with torch.no_grad():
        if use_cuda:
            data, label = data.cuda(), label.cuda()
        outputs = net(data)
    acc1 = accuracy(outputs, label, topk=(1,))
    metric.update(acc1[0], data.size(0))
    return acc1[0]

def default_train_fn(net, data, label, batch_idx, epoch, criterion,
                     optimizer, lr_scheduler, use_cuda):
    net.train()
    lr_scheduler(optimizer, batch_idx, epoch)
    optimizer.zero_grad()
    if use_cuda:
        data, label = data.cuda(), label.cuda()
    outputs = net(data)
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step()
