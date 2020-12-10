import autogluon.core as ag
from autogluon.vision.object_detection import ObjectDetector
import os

url = 'https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip'
dataset_train = ObjectDetector.Dataset.from_voc(url, splits='train')

time_limit = 5*60*60 # 5 hours
epochs = 30
detector = ObjectDetector()
detector.fit(dataset_train,
             num_trials=2,
             epochs=epochs,
             hyperparameters={'lr': ag.Categorical(5e-4, 1e-4)},
             ngpus_per_trial=1,
             time_limit=time_limit)

# Evaluation on test dataset
dataset_test = ObjectDetector.Dataset(url, splits='test')
test_map = detector.evaluate(dataset_test)
print("mAP on test dataset: {}".format(test_map[1][1]))

# visualization
image = '000467.jpg'
image_path = os.path.join(data_root, 'JPEGImages', image)
print(image_path)

result = detector.predict(image_path)
print('Prediction result:', result)
