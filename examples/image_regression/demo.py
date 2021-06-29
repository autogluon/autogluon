import autogluon.core as ag
from autogluon.vision import ImagePredictor
import os


IMAGE_REGRESS_TRAIN, _, IMAGE_REGRESS_TEST = ImagePredictor.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')

# Train
predictor = ImagePredictor(problem_type='regression')
predictor.fit(train_data=IMAGE_REGRESS_TRAIN, hyperparameters={'epochs': 3, 'batch_size': 8})

# Evaluation on test dataset
test_result = predictor.predict(IMAGE_REGRESS_TEST)
print('test_result:')
print(test_result)

test_evaluation = predictor.evaluate(IMAGE_REGRESS_TEST)
print('Evaluation result:', test_evaluation)

result = predictor.predict(IMAGE_REGRESS_TEST.iloc[0]['image'])
print('Prediction result for single image:')
print(result)
