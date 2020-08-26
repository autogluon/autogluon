# FAQ
:label:`sec_faq`

How can I perform inference on a file that won't fit in memory?

The Tabular Dataset API works with pandas Dataframes, which supports chunking data into sizes that fit in memory.
Here's an example of one such chunk-based inference:

```{.python .input}
from autogluon import TabularPrediction as task
import pandas as pd
import requests

train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
predictor = task.fit(train_data=train_data.sample(n=100, random_state=0), label='class', hyperparameters={'GBM': {}})

# Get the test dataset, if you are working with local data then omit the next two lines
r = requests.get('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv', allow_redirects=True)
open('test.csv', 'wb').write(r.content)
reader = pd.read_csv('test.csv', chunksize=1024)
y_pred = []
y_true = []
for df_chunk in reader:
    y_pred.append(predictor.predict(df_chunk, as_pandas=True))
    y_true.append(df_chunk['class'])
y_pred = pd.concat(y_pred, axis=0, ignore_index=True)
y_true = pd.concat(y_true, axis=0, ignore_index=True)
predictor.evaluate_predictions(y_true=y_true, y_pred=y_pred)
```

Here we split the test data into chunks of up to 1024 rows each, but you may select a larger size as long as it fits into your system's memory.
[Further Reading](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-chunking)
