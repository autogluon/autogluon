from autogluon import image_classification as task
dataset = task.Dataset(name='shopeeiet', train_path='~/data/train')

time_limits = 1 * 60 # 10mins
num_training_epochs = 1
results = task.fit(dataset, time_limits=time_limits, num_training_epochs=num_training_epochs)

#
inds, probs = task.predict_batch('/home/ubuntu/data/test/BabyPants/')

import autogluon as ag
ag.utils.generate_csv(inds, '/home/ubuntu/data/submission.csv')

dataset = task.Dataset(name='shopeeiet', train_path='~/data/train')

time_limits = 1*60
num_training_epochs = 10
results = task.fit(dataset,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)

print('Top-1 val acc: %.3f' % results.metric)
test_dataset = task.Dataset(name='shopeeiet', test_path='~/data/test')
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)

import autogluon as ag
net_list = ['resnet18_v1',
            'resnet34_v1',
            'resnet50_v1']

# default net list for image classification would be overwritten
# if net_list is provided
nets = ag.Nets(net_list)

print(nets)

results = task.fit(dataset,
                   nets,
                   time_limits=time_limits,
                   num_training_epochs=num_training_epochs)
print('Top-1 val acc: %.3f' % results.metric)
test_acc = task.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc)
