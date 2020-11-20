"""
This example reproduces the ResNet CIFAR10 experiment from https://arxiv.org/abs/2003.10865.
To run the a ResNet on CIFAR10, we bootstrapped from https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469

We use pytorch to implement the neural network. To install it, follow the steps described here:
https://pytorch.org/get-started/locally/.
"""

import autogluon.core as ag
import time
import logging
import yaml
import argparse
import numpy as np
import torch

from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

from torchvision import models, datasets
import torchvision.transforms as transforms


def get_CIFAR10(root="./"):
    input_size = 32
    num_classes = 10

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=train_transform, download=True
    )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    valid_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=valid_transform, download=True
    )

    return input_size, num_classes, train_dataset, valid_dataset


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=10)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x


def train(model, train_loader, optimizer, epoch):
    model.train()

    total_loss = []

    for data, target in train_loader:
        if torch.cuda.is_available():

            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()

        prediction = model(data)
        loss = F.nll_loss(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")


def valid(model, valid_loader):
    model.eval()

    loss = 0
    correct = 0

    for data, target in valid_loader:
        with torch.no_grad():
            if torch.cuda.is_available():

                data = data.cuda()
                target = target.cuda()

            prediction = model(data)
            loss += F.nll_loss(prediction, target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(valid_loader.sampler)

    percentage_correct = 100.0 * correct / len(valid_loader.sampler)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(valid_loader.sampler), percentage_correct
        )
    )

    return loss, percentage_correct / 100


def make_res_net_benchmark(dataset_path, num_gpus=1):
    @ag.args(
        batch_size=ag.space.Int(lower=8, upper=256),
        momentum=ag.space.Real(lower=0, upper=.99),
        weight_decay=ag.space.Real(lower=1e-5, upper=1e-3, log=True),
        lr=ag.space.Real(lower=1e-3, upper=1e-1, log=True),
        dataset_path=dataset_path,
        num_gpus=num_gpus,
        epochs=27)
    def objective_function(args, reporter, **kwargs):

        ts_start = time.time()

        torch.manual_seed(np.random.randint(10000))

        batch_size = args.batch_size
        lr = args.lr
        momentum = args.momentum
        weight_decay = args.weight_decay

        input_size, num_classes, train_dataset, valid_dataset = get_CIFAR10(root=args.dataset_path)

        indices = list(range(train_dataset.data.shape[0]))
        train_idx, valid_idx = indices[:40000], indices[40000:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=0,
                                                   sampler=train_sampler,
                                                   pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=5000,
                                                   num_workers=0,
                                                   sampler=valid_sampler,
                                                   pin_memory=True)

        model = Model()
        if torch.cuda.is_available():
            model = model.cuda()
            device = torch.device("cuda")
            model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.num_gpus)]).to(device)

        milestones = [25, 40]

        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )

        for epoch in range(1, args.epochs + 1):

            train(model, train_loader, optimizer, epoch)
            loss, y = valid(model, valid_loader)
            scheduler.step()

            config = dict()
            config["batch_size"] = batch_size
            config["lr"] = lr
            config["momentum"] = momentum
            config["weight_decay"] = weight_decay

            ts_now = time.time()
            eval_time = ts_start - ts_now
            reporter(
                epoch=epoch,
                performance=y,
                eval_time=eval_time,
                time_step=ts_now, **config)
    return objective_function


def callback(training_history, start_timestamp):
    # This callback function will be executed every time AutoGluon collected some new data.
    # In this example we will parse the training history into a .csv file and save it to disk, such that we can
    # for example plot AutoGluon's performance during the optimization process.
    # If you don't care about analyzing performance of AutoGluon online, you can also ignore this callback function and
    # just save the training history after AutoGluon has finished.
    import pandas as pd
    task_dfs = []

    # this function need to be changed if you return something else than accuracy
    def compute_error(df):
        return 1.0 - df["performance"]

    def compute_runtime(df, start_timestamp):
        return df["time_step"] - start_timestamp

    for task_id in training_history:
        task_df = pd.DataFrame(training_history[task_id])
        task_df = task_df.assign(task_id=task_id,
                                 runtime=compute_runtime(task_df, start_timestamp),
                                 error=compute_error(task_df),
                                 target_epoch=task_df["epoch"].iloc[-1])
        task_dfs.append(task_df)

    result = pd.concat(task_dfs, axis="index", ignore_index=True, sort=True)

    # re-order by runtime
    result = result.sort_values(by="runtime")

    # calculate incumbent best -- the cumulative minimum of the error.
    result = result.assign(best=result["error"].cummin())

    result.to_csv("history.csv")


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='runs autogluon on autoff benchmarks')

    parser.add_argument('--num_trials', default=10, type=int,
                        help='number of trial tasks')
    parser.add_argument('--timeout', default=1000, type=int,
                        help='runtime of autogluon in seconds')
    parser.add_argument('--num_gpus', type=int, default=0,
                        help='number of GPUs available to a given trial.')
    parser.add_argument('--num_cpus', type=int, default=4,
                        help='number of CPUs available to a given trial.')
    parser.add_argument('--hostfile', type=argparse.FileType('r'))
    parser.add_argument('--store_results_period', type=int, default=100,
                        help='If specified, results are stored in intervals of '
                             'this many seconds (they are always stored at '
                             'the end)')
    parser.add_argument('--scheduler', type=str, default='hyperband_promotion',
                        choices=['hyperband_stopping', 'hyperband_promotion'],
                        help='Asynchronous scheduler type. In case of doubt leave it to the default')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    logging.root.setLevel(logging.INFO)

    # In case you want to run AutoGluon across multiple instances, you need to provide it with a list of
    # the IP addresses of the instances. Here, we assume that the IP addresses are stored in a .yaml file.
    # If you want to run AutoGluon on a single instance just pass None.
    # However, keep in mind that it will still parallelize the optimization process across multiple threads then.
    # If you really want to run it purely sequentially, set the num_cpus equal to the number of VCPUs of the machine.
    dist_ip_addrs = yaml.load(args.hostfile) if args.hostfile is not None else []

    del args.hostfile

    print("Got worker host IP addresses [{}]".format(dist_ip_addrs))

    if args.scheduler == "hyperband_stopping":
        hyperband_type = "stopping"

    elif args.scheduler == "hyperband_promotion":
        hyperband_type = "promotion"

    brackets = 1  # setting the number of brackets to 1 means that we run effectively successive halving
    func = make_res_net_benchmark('./cifar10_dataset')
    scheduler = ag.scheduler.HyperbandScheduler(func,
                                                resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
                                                # Autogluon runs until it either reaches num_trials or time_out
                                                num_trials=args.num_trials,
                                                time_out=args.timeout,
                                                # This argument defines the metric that will be maximized.
                                                # Make sure that you report this back in the objective function.
                                                reward_attr='performance',
                                                # The metric along we make scheduling decision. Needs to be also
                                                # reported back to AutoGluon in the objective function.
                                                time_attr='epoch',
                                                brackets=brackets,
                                                checkpoint=None,
                                                searcher="bayesopt",  # Defines searcher for new configurations
                                                dist_ip_addrs=dist_ip_addrs,
                                                training_history_callback=callback,
                                                training_history_callback_delta_secs=args.store_results_period,
                                                reduction_factor=3,
                                                type=hyperband_type,
                                                # defines the minimum resource level for Hyperband,
                                                # i.e the minimum number of epochs
                                                grace_period=1
                                                )
    scheduler.run()
    scheduler.join_jobs()
