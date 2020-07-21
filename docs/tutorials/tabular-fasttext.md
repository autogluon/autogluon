# fastText introduction

fastText model is a simple and fast baseline model for text classification. It learns about features (n-grams)  embedding, which are averaged to form the hideen vector representation of a document. Its accuray is on par with deep learning classifiers, but is orders of magnititute faster for training and evaluation. The fastYext model provides another baseline model for text classification besides bags of words model in autogluon.   

To start, import autogluon and TabularPrediction module as your task:


```python
import logging
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
```


```python
import pandas as pd
import autogluon as ag
from autogluon import TabularPrediction as task
from autogluon.utils.tabular.ml.models.fasttext.fasttext_model import FastTextModel
from autogluon.utils.tabular.ml.utils import infer_problem_type
from autogluon.task.tabular_prediction.hyperparameter_configs import get_hyperparameter_config
```

# load data

Load training data from a CSV file into an AutoGluon Dataset object. This object is essentially equivalent to a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) and the same methods can be applied to both.


```python
train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
train_data['class'] = train_data['class'].str.strip()
test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
print(train_data.head())
```

Note that we loaded data from a CSV file stored in the cloud (AWS s3 bucket), but you can you specify a local file-path instead if you have already downloaded the CSV file to your own machine (e.g., using `wget`).
Each row in the table `train_data` corresponds to a single training example. In this particular dataset, each row corresponds to an individual person, and the columns contain various characteristics reported during a census.

Let's first use these features to predict whether the person's income exceeds $50,000 or not, which is recorded in the `class` column of this table.


```python
label_column = 'class'
print("Summary of class variable: \n", train_data[label_column].describe())
```

# Train FastText model alone

The FastTextModel wrapper in AutoGluon can be used on its own, which provides a convenient interface for working with typical tabular data so that you do not need to handle the format conversion as required by the vanilla fasttext model implementation. Note the all options accepted by the original fasttext model can be specified through the hyperparameters option. 


```python
X_train = train_data.drop(columns=[label_column])
y_train = train_data[label_column]

problem_type = infer_problem_type(y=y_train)  # Infer problem type (or else specify directly)

fasttext_model = FastTextModel(path='fasttext-model/', name='FastTextModel', problem_type=problem_type,
                              hyperparameters={'epoch': 50})
fasttext_model.fit(X_train=X_train, y_train=y_train)
```

Now let's see check the model performance on the test data:


```python
y_pred = fasttext_model.predict(test_data)

df_res = pd.DataFrame({
    'pred': y_pred,
    'label': test_data[label_column]
})
print('accuracy:', (df_res.pred.str.strip() == df_res.label.str.strip()).mean())
print(df_res.sample(5))
```

# Use fastText model in TabularPrediction task 

Let's add some mock text fields to the original data


```python
train_data['text'] = (
    train_data[['education', 'marital-status', 'occupation', 'relationship', 
                'workclass', 'native-country',  'sex', 'race']]
    .apply(lambda r: ', '.join(r.values) + '.', axis=1)
)


test_data['text'] = (
    test_data[['education', 'marital-status', 'occupation', 'relationship',
               'workclass', 'native-country',  'sex', 'race']]
    .apply(lambda r: ', '.join(r.values) + '.', axis=1)
)
print('sample text column values')
print(train_data['text'].sample(5).to_list())
```

Now, we can specific FastTextModel as one custom model so that you can leverage the emsemble/stacking feature in AutoGluon:


```python
custom_hyperparameters = {'RF': {},
                         FastTextModel:  {'epoch': 50},
                         }

predictor = task.fit(train_data=train_data, 
                     label=label_column, 
                     hyperparameters=custom_hyperparameters
                    )

y_pred = predictor.predict(test_data)
df_res = pd.DataFrame({
    'pred': y_pred,
    'label': test_data[label_column]
})
print('accuracy:', (df_res.pred.str.strip() == df_res.label.str.strip()).mean())
print(df_res.sample(5))
```
