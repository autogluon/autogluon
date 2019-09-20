import logging
logging.basicConfig(level=logging.INFO)

from autogluon import image_classification as task
dataset = task.Dataset(name='shopeeiet', train_path='~/data/shopeeiet/train', val_path='~/data/shopeeiet/val')

time_limits = 1 * 60 # 10mins
num_training_epochs = 1
results = task.fit(dataset, time_limits=time_limits, num_training_epochs=num_training_epochs)

#
inds, probs = task.predict_batch('~/data/shopeeiet/test')

import autogluon as ag
ag.utils.generate_csv(inds, '~/data/shopeeiet/submission.csv')

print('finished!')

# print('Top-1 val acc: %.3f' % results.metric)
# test_acc = task.evaluate(test_dataset)
# print('Top-1 test acc: %.3f' % test_acc)

# import os
# os.system('wget http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/shopeeiet/data_shopeeiet.zip')
# os.system('unzip -o data_shopeeiet.zip -d ~/')

# from autogluon import image_classification as task
# dataset = task.Dataset(name='shopeeiet', train_path='~/data/shopeeiet/train')
# #
# time_limits = 1*60
# num_training_epochs = 10
# #
# results = task.fit(dataset,
#                    searcher='random',
#                    time_limits=time_limits,
#                    num_training_epochs=num_training_epochs)
# #
# inds, probs = task.predict_batch('~/data/shopeeiet/test')
#
# import autogluon as ag
# ag.utils.generate_csv(inds, '~/data/shopeeiet/submission.csv')
#
#
# # from autogluon import image_classification as task
#
# # import os
# # os.system('wget http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/data.zip')
# # os.system('unzip -o data.zip -d ~/')
#
# dataset = task.Dataset(name='shopeeiet', train_path='~/data/train')
#
# # dataset = task.Dataset(name='apparel', train_path='/home/ubuntu/data/apparel/Apparel_train.rec')
#
# # time_limits = 1*60
# # num_training_epochs = 10
#
# results = task.fit(dataset,
#                    searcher='bayesopt',
#                    time_limits=time_limits,
#                    num_training_epochs=num_training_epochs)
print('all finished!')
#
# # import autogluon as ag
# # results = task.fit(dataset,
# #                    searcher=searcher,
# #                    time_limits=time_limits,
# #                    num_training_epochs=num_training_epochs,
# #                    lr_scheduler=ag.space.List('lr_scheduler',
# #                                               ['mx.lr_scheduler.PolyScheduler',
# #                                                'mx.lr_scheduler.CosineScheduler']))
# #
# print('Top-1 val acc: %.3f' % results.metric)
# test_dataset = task.Dataset(name='shopeeiet', test_path='/home/ubuntu/data/apparel/Apparel_test.rec')
# test_acc = task.evaluate(test_dataset)
# print('Top-1 test acc: %.3f' % test_acc)
# #
# # print('Default models:')
# # print(results.metadata['nets'])
# #
# # print('Default optimizers:')
# # print(results.metadata['optimizers'])
# #
# # print('Default searcher:')
# # print(results.metadata['searcher'])
# #
# # print('Default trial scheduler:')
# # print(results.metadata['trial_scheduler'])
# #
# # print('Stop criterion:')
# # print(results.metadata['stop_criterion'])
# #
# # print('Resources used for each trial:')
# # print(results.metadata['resources_per_trial'])
# #
# # image = '/home/ubuntu/data/test/BabyShirt/BabyShirt_323.jpg'
# # ind, prob = task.predict(image)
# # print('The input picture is classified as [%s], with probability %.2f.' %
# #       (dataset.train.synsets[ind.asscalar()], prob.asscalar()))
# #
# # # # task.trial_scheduler.shutdown()
# # # print('1 fit finished!')
# # #
# # # import autogluon as ag
# # # net_list = ['resnet18_v1',
# # #             'resnet34_v1',
# # #             'resnet50_v1']
# # #
# # # # default net list for image classification would be overwritten
# # # # if net_list is provided
# # # nets = ag.Nets(net_list)
# # #
# # # # print(nets)
# # # #
# # # # results = task.fit(dataset,
# # # #                    nets,
# # # #                    time_limits=time_limits,
# # # #                    num_training_epochs=num_training_epochs)
# # # #
# # # # print('Top-1 val acc: %.3f' % results.metric)
# # # # test_acc = task.evaluate(test_dataset)
# # # # print('Top-1 test acc: %.3f' % test_acc)
# # # #
# # # # # task.trial_scheduler.shutdown()
# # # # print('2 fit finished!')
# # # #
# # # #
# # # # optimizers_default = ag.Optimizers(['sgd', 'adam'])
# # #
# # # adam_opt = ag.optimizers.Adam(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
# # #                           wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
# # # sgd_opt = ag.optimizers.SGD(lr=ag.space.Log('lr', 10 ** -4, 10 ** -1),
# # #                         momentum=ag.space.Linear('momentum', 0.85, 0.95),
# # #                         wd=ag.space.Log('wd', 10 ** -6, 10 ** -2))
# # # optimizers = ag.Optimizers([adam_opt, sgd_opt])
# # #
# # # # print(optimizers)
# # # #
# # # # results = task.fit(dataset,
# # # #                    nets,
# # # #                    optimizers,
# # # #                    time_limits=time_limits,
# # # #                    num_training_epochs=num_training_epochs)
# # # #
# # # # print('Top-1 val acc: %.3f' % results.metric)
# # # # test_acc = task.evaluate(test_dataset)
# # # # print('Top-1 test acc: %.3f' % test_acc)
# # # #
# # # # # task.trial_scheduler.shutdown()
# # # # print('3 fit finished!')
# # #
# # # searcher = 'random'
# # #
# # # trial_scheduler_fifo = 'fifo'
# # # trial_scheduler = 'hyperband'
# # #
# # # results = task.fit(dataset,
# # #                    nets,
# # #                    optimizers,
# # #                    searcher=searcher,
# # #                    trial_scheduler=trial_scheduler,
# # #                    time_limits=time_limits,
# # #                    num_training_epochs=num_training_epochs)
# # #
# # # print('Top-1 val acc: %.3f' % results.metric)
# # # test_acc = task.evaluate(test_dataset)
# # # print('Top-1 test acc: %.3f' % test_acc)
# # #
# # # print('4 fit finished!')
# # #
# # # image = '/home/ubuntu/data/test/BabyShirt/BabyShirt_323.jpg'
# # # ind, prob = task.predict(image)
# # # print('The input picture is classified as [%s], with probability %.2f.' %
# # #       (dataset.train.synsets[ind.asscalar()], prob.asscalar()))
# # #
# # # # task.trial_scheduler.shutdown()
# # # print('all finished!')