# Text Prediction - Mixed Data Type
:label:`sec_textprediction_heterogeneous`

For real-world applications, text data are usually mixed with other common data types like 
numerical data and categorical data. Here, the `TextPrediction` task in AutoGluon 
handles the mix of multiple feature types, including text, categorical, and numerical. 
Next, we will use the Semantic Textual Segmentation dataset that we have used to illustrate 
this functionality.


```{.python .input}
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Load Data

```{.python .input}
from autogluon.utils.tabular.utils.loaders import load_pd

train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet')
dev_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet')
train_data.head(10)
```

We can see that the STS dataset contains two text fields: `sentence1` and `sentence2`, one categorical field: `genre`, and one numerical field `score`. 
We try to predict the score with `sentence1` + `sentence2` + `genre`.


```{.python .input}
import autogluon as ag
from autogluon import TextPrediction as task

predictor_score = task.fit(train_data, label='score',
                           time_limits=60, ngpus_per_trial=1, seed=123,
                           output_directory='./ag_sts_mixed_score')
```


```{.python .input}
score = predictor_score.evaluate(dev_data, metrics='spearmanr')
print('Spearman Correlation=', score['spearmanr'])
```

In addition, we can also train a model that predicts the `genre` with the other columns


```{.python .input}
predictor_genre = task.fit(train_data, label='genre',
                           time_limits=60, ngpus_per_trial=1, seed=123,
                           output_directory='./ag_sts_mixed_genre')
```


```{.python .input}
score = predictor_genre.evaluate(dev_data, metrics='acc')
print('Genre Accuracy = {}%'.format(score['acc'] * 100))
```
