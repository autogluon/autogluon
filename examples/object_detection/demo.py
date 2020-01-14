import autogluon as ag
from autogluon import ObjectDetection as task
import os

root = './'
filename_zip = ag.download('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip',
                        path=root)
filename = ag.unzip(filename_zip, root=root)


data_root = os.path.join(root, filename)
dataset_train = task.Dataset(data_root, classes=('motorbike',))

time_limits = 5*60*60 # 5 hours
epochs = 30
detector = task.fit(dataset_train,
                    num_trials=2,
                    epochs=epochs,
                    lr=ag.Categorical(5e-4, 1e-4),
                    ngpus_per_trial=1,
                    time_limits=time_limits)

# Evaluation on test dataset
dataset_test = task.Dataset(data_root, index_file_name='test', classes=('motorbike',))
test_map = detector.evaluate(dataset_test)
print("mAP on test dataset: {}".format(test_map[1][1]))

# visualization 
image = '000467.jpg'
image_path = os.path.join(data_root, 'JPEGImages', image)
print(image_path)

ind, prob, loc = detector.predict(image_path)
