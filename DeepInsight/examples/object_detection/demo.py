import autogluon.core as ag
from autogluon.vision.object_detection import ObjectDetector
import os

url = 'https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip'
dataset_train = ObjectDetector.Dataset.from_voc(url, splits='trainval')

time_limit = 5*60*60 # 5 hours
epochs = 3
detector = ObjectDetector()
detector.fit(dataset_train,
             num_trials=2,
             hyperparameters={'lr': ag.Categorical(5e-4, 1e-4), 'epochs': epochs},
             ngpus_per_trial=1,
             time_limit=time_limit)

# Evaluation on test dataset
dataset_test = ObjectDetector.Dataset.from_voc(url, splits='test')
test_map = detector.evaluate(dataset_test)
print("mAP on test dataset: {}".format(test_map[1][1]))

# visualization
result = detector.predict(dataset_test.iloc[0]['image'])
print('Prediction result:', result)
