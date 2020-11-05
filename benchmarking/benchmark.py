import torch
import torch.nn as nn
import torch.nn.functional as F
import autogluon.core as ag

import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import argparse
from datetime import datetime
import wandb

parser = argparse.ArgumentParser()

parser.add_argument('--num_cpus', default=6, type=int, help='number of CPUs to use')
parser.add_argument('--num_gpus', default=0, type=int, help='number of GPUs to use')
parser.add_argument('--num_trials', default=6, type=int, help='number of trials to run')
parser.add_argument('--ip', default=None, help='additional ips to be added')

args = parser.parse_args()


class ConvNet(nn.Module):
    def __init__(self, hidden_conv, hidden_fc):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_conv, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(hidden_conv, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_data_loaders(batch_size):
    """
    batch_size: The batch size of the dataset

    Returns the train and test data loaders
    """

    # transforms the the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load the datasets
    train_data = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # create the data loaders
    train_data = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False
    )
    test_data = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_data, test_data


# Hyper Parameters to search over
@ag.args(
    lr=ag.space.Categorical(0.01, 0.2),
    wd=ag.space.Categorical(1e-4, 5e-4),
    epochs=ag.space.Categorical(5, 6),
    hidden_conv=ag.space.Categorical(6, 7),
    hidden_fc=ag.space.Categorical(80, 120),
    batch_size=ag.space.Categorical(16, 32, 64, 128)
)
def train_image_classification(args, reporter):
    """
    args: arguments passed to the function through the ag.args designator
    reporter: The aug reporter object passed to the function by autogluon

    Reports the accuracy of the model to be monitored
    """

    # get variables from args
    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    batch_size = args.batch_size
    model = ConvNet(args.hidden_conv, args.hidden_fc)

    # get the data loaders
    train_loader, test_loader = get_data_loaders(batch_size)

    # check if gpu is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # if multiple GPUs are available, make it a data parallel model
    if device == 'cuda':
        model = nn.DataParallel(model)

    # get the loss function, and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=wd)

    # def the train function
    def train():
        """
        Trains the model
        """

        # set the model to train mode
        model.train()

        # run through all the batches in the dataset
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # move the data to the target device
            inputs, targets = inputs.to(device), targets.to(device)

            # zero out gradients
            optimizer.zero_grad()

            # forward pass through the network
            outputs = model(inputs)

            # calculate loss and backward pass
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    def test(epoch):
        """
        epoch: epoch number

        Tests the model
        """

        # set the model to evaluation mode
        model.eval()

        # keep track of the test loss and correct predcitons
        test_loss, correct, total = 0, 0, 0

        # stop tracking the gradients, reduces memory consumption
        with torch.no_grad():
            # run through the test set
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                # move the inputs to the target device
                inputs, targets = inputs.to(device), targets.to(device)

                # forward pass through the network
                outputs = model(inputs)

                # calculate the loss and labels
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # keep track of thethe total correct predicitons
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # calculate the accuracy
        acc = 100. * correct / total

        # report the accuracy and the parameters used
        reporter(epoch=epoch, accuracy=acc, lr=lr, wd=wd, batch_size=batch_size)

    # run the testing and training script
    for epoch in tqdm(range(0, epochs)):
        train()
        test(epoch)


# define all the tasks
tasks = [
    train_image_classification,  # image classification task
]

# define run time table
run_times = []
ext_ips = [args.ip]

# Run every task with all available schedulers
for task in tasks:

    # define all schedulers
    schedulers = [
        ag.scheduler.FIFOScheduler(
            task,
            resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
            num_trials=args.num_trials,
            time_attr='epoch',
            reward_attr='accuracy',
             dist_ip_addrs=ext_ips
        ),  # add the FIFO scheduler

        ag.scheduler.HyperbandScheduler(
            task,
            resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
            num_trials=args.num_trials,
            time_attr='epoch',
            reward_attr='accuracy',
            dist_ip_addrs=ext_ips
        ),  # add the Hyperband scheduler

        ag.scheduler.RLScheduler(
            task,
            resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
            num_trials=args.num_trials,
            time_attr='epoch',
            reward_attr='accuracy',
            dist_ip_addrs = ext_ips
        )  # add the FIFO scheduler
    ]

    # define the scheduler run time list
    scheduler_runtimes = []

    # run the task with each scheduler
    for scheduler in schedulers:
        # initialize the logging
        run = wandb.init(project='autogluon')

        # display the scheduler and available resources
        print('')
        print(scheduler)
        print('')

        # start the clock
        start_time = datetime.now()

        # run the job with the scheduler
        scheduler.run()
        scheduler.join_jobs()

        # stop the clock
        stop_time = datetime.now()
        scheduler_runtimes.append((stop_time - start_time).total_seconds())

        # end the run
        run.finish()

    run_times.append(scheduler_runtimes)

print(run_times)
