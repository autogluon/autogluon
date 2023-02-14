# AutoMM Presets
:label:`sec_automm_presets`

It is well-known that we usually need to set hyperparameters before the learning process begins. Deep learning models, e.g., pretrained foundation models, can have anywhere from a few hyperparameters to a few hundred. The hyperparameters can impact training speed, final model performance, and inference latency. However, choosing the proper hyperparameters may be challenging for many users with limited expertise.

In this tutorial, we will introduce the easy-to-use presets in AutoMM. Our presets can condense the complex hyperparameter setups into simple strings. More specifically, AutoMM supports three presets: `medium_quality`, `high_quality`, and `best_quality`.


```{.python .input}
import warnings
warnings.filterwarnings('ignore')
```

## Dataset

For demonstration, we use a subsampled Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)) dataset, which consists of movie reviews and their associated sentiment. 
Given a new movie review, the goal is to predict the sentiment reflected in the text (in this case, a **binary classification**, where reviews are 
labeled as 1 if they conveyed a positive opinion and 0 otherwise).
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
In some situations, we prefer fast training and inference to the prediction quality. `medium_quality` is designed for this purpose.
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
If you want to balance the prediction quality and training/inference speed, you can try the `high_quality` preset, which uses a larger model than `medium_quality`. Accordingly, we need to increase the time limit since larger models require more time to train.


```{.python .input}
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label='label', eval_metric='acc', presets="high_quality")
predictor.fit(
    train_data=train_data,
    time_limit=50, # seconds
)
```

Although `high_quality` requires more training time than `medium_quality`, it also brings performance gains.


```{.python .input}
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
scores
```

## Best Quality
If you want the best performance and don't care about the training/inference cost, give it a try for the `best_quality` preset. High-end GPUs with large memory are preferred in this case. Compared to `high_quality`, it requires much longer training time.


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
The above three presets all use the default hyperparameters, which might not be optimal for your tasks. Fortunately, we also support hyperparameter optimization (HPO) with simple presets. To perform HPO, you can add a postfix `_hpo` in the three presets, resulting in `medium_quality_hpo`, `high_quality_hpo`, and `best_quality_hpo`.

## Display Presets
In case you want to see each preset's inside details, we provide you with a util function to get the hyperparameter setups. For example, here are hyperparameters of preset `high_quality`. 


```{.python .input}
import json
from autogluon.multimodal.presets import get_automm_presets
hyperparameters, hyperparameter_tune_kwargs = get_automm_presets(problem_type="default", presets="high_quality")
print(f"hyperparameters: {json.dumps(hyperparameters, sort_keys=True, indent=4)}")
print(f"hyperparameter_tune_kwargs: {json.dumps(hyperparameter_tune_kwargs, sort_keys=True, indent=4)}")
```

The HPO presets make several hyperparameters tunable such as model backbone, batch size, learning rate, max epoch, and optimizer type. Below are the details of preset `high_quality_hpo`.


```{.python .input}
import json
import yaml
from autogluon.multimodal.presets import get_automm_presets
hyperparameters, hyperparameter_tune_kwargs = get_automm_presets(problem_type="default", presets="high_quality_hpo")
print(f"hyperparameters: {yaml.dump(hyperparameters, allow_unicode=True, default_flow_style=False)}")
print(f"hyperparameter_tune_kwargs: {json.dumps(hyperparameter_tune_kwargs, sort_keys=True, indent=4)}")
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
