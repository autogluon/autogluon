# AutoMM Presets
:label:`sec_automm_presets`

It is well known that we usually need to set hyperparameters before the learning process begins. Deep learning models, e.g., pretrained foundation models, can have anywhere from a few hyperparameters to a few hundred hyperparameters. The hyperparameters can impact the learning rate, final model performance, as well as the inference speed. However, choosing the proper hyperparameters may be challenging for many users with limited expertise. 

In this tutorial, we will introduce the easy-to-use presets in AutoMM. Our presets condense the complex hyperparameter setups into simple strings. More specifically, AutoMM supports three presets: `medium_quality`, `high_quality`, and `best_quality`.


```{.python .input}
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset

For demonstration, we use a subsampled Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)) dataset, which consists of movie reviews and their associated sentiment. 
Given a new movie review, the goal is to predict the sentiment reflected in the text (in this case a **binary classification**, where reviews are 
labeled as 1 if they convey a positive opinion and labeled as 0 otherwise).
To get started, let's download and prepare the dataset.


```{.python .input}
from autogluon.core.utils.loaders import load_pd
train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
subsample_size = 1000  # subsample data for faster demo, try setting this to larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head(10)
```

## Medium Quality
In some situations, we prefer fast training and inference over the prediction quality. `medium_quality` is designed for this purpose.
Among the three presets, `medium_quality` has the smallest model size. Now let's fit the predictor using the `medium_quality` preset. Here we set a tight time budget for a quick demo.


```{.python .input}
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="medium_quality")
predictor.fit(
    train_data=train_data,
    time_limit=30, # seconds
)
```

Then we can evaluate the predictor on the test data.


```{.python .input}
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
scores
```

## High Quality
If you want to balance the prediction quality and training/inference speed, you can try the `high_quality` preset, which uses a larger model than `medium_quality`. Accordingly, we need to increase the time limit since larger models requires more time to train.


```{.python .input}
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="high_quality")
predictor.fit(
    train_data=train_data,
    time_limit=60, # seconds
)
```

Although `high_quality` requires more training time than `medium_quality`, it also brings performance gains.


```{.python .input}
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
scores
```

## Best Quality
If you want the best performance and don't care about the training/inference cost, give it a try about the `best_quality` preset . High-end GPUs with large memory is preferred in this case. Compared to `high_quality`, it requires much longer training time.


```{.python .input}
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="best_quality")
predictor.fit(train_data=train_data, time_limit=180)
```

We can see that `best_quality` achieves better performance than `high_quality`.


```{.python .input}
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
scores
```

## HPO Presets
The above three presets all use the default hyperparameters, which might not be optimal for your tasks. Fortunately, we also support doing hyperparameter optimization (HPO) with simple presets. To perform HPO, you can simply add a posfix `_hpo` in the three presets, resulting in `medium_quality_hpo`, `high_quality_hpo`, and `best_quality_hpo`.

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
