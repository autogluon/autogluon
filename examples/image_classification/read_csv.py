import pandas as pd
import os

# kaggle = 'dogs-vs-cats-redux-kernels-edition/'
# kaggle = 'aerial-cactus-identification/'
kaggle = 'plant-seedlings-classification/'
# kaggle = 'fisheries_Monitoring/'
# kaggle = 'dog-breed-identification/'
# kaggle = 'shopee-iet-machine-learning-competition/'

csv_path = os.path.join('/home/ubuntu/workspace/data/dataset/', kaggle,'sample_submission.csv')
# csv_path = os.path.join('/home/ubuntu/workspace/train_script/autogluon_1208/examples/image_classification/, kaggle,'predict.csv')

df = pd.read_csv(csv_path)

print(df.head)