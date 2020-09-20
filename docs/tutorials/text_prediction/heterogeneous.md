# Text Prediction - Heterogeneous Data Types
:label:`sec_textprediction_heterogeneous`

In your applications, your text data may be mixed with other common data types like 
numerical data and categorical data (which are commonly found in tabular data). The `TextPrediction` task in AutoGluon 
can train a single neural network that jointly operates on multiple feature types, including text, categorical, and numerical columns. 
Here we'll again use the [Semantic Textual Similarity](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) dataset to illustrate 
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

Note the STS dataset contains two text fields: `sentence1` and `sentence2`, one categorical field: `genre`, and one numerical field `score`. 
Let's try to predict the **score** based on the other features: `sentence1`, `sentence2`, `genre`.


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

We can also train a model that predicts the **genre** using the other columns as features.


```{.python .input}
predictor_genre = task.fit(train_data, label='genre',
                           time_limits=60, ngpus_per_trial=1, seed=123,
                           output_directory='./ag_sts_mixed_genre')
```


```{.python .input}
score = predictor_genre.evaluate(dev_data, metrics='acc')
print('Genre-prediction Accuracy = {}%'.format(score['acc'] * 100))
```
