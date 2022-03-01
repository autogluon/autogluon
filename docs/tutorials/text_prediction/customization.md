# Text Prediction - Customization
:label:`sec_textprediction_customization`

This tutorial introduces how to customize the hyperparameters in `TextPredictor`.


```python
import numpy as np
import warnings
import autogluon as ag
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Stanford Sentiment Treebank Data

For demonstration, we use the Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)) dataset.


```python
from autogluon.core import TabularDataset
subsample_size = 1000  # subsample for faster demo, you may try specifying larger value
train_data = TabularDataset('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet')
test_data = TabularDataset('https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet')
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head(10)
```

## Configure TextPredictor

### Preset Configurations

AutoMM provides several simple preset configurations. Let's take a look at the available presets.


```python
from autogluon.text.text_prediction.presets import list_text_presets
list_text_presets()
```

You may be interested in the configuration differences behind the preset strings.


```python
list_text_presets(verbose=True)
```

We can find that the main difference between different presets lie in the choices of huggingface transformer checkpoints. Preset `default` has the same configuration as preset `high_quality`. Designing the presets follows the rule of thumb that larger backbones tend to have better performance but with higher cost.

Let's train a text predictor with preset `medium_quality_faster_train`. 


```python
from autogluon.text import TextPredictor
predictor = TextPredictor(eval_metric='acc', label='label')
predictor.fit(
    train_data=train_data,
    presets='medium_quality_faster_train',
    time_limit=60,
)
```

Below we report both `f1` and `acc` metrics for our predictions. If you want to obtain the best F1 score, you should set `eval_metric='f1'` when constructing the TextPredictor.


```python
predictor.evaluate(test_data, metrics=['f1', 'acc'])
```

### Custom Hyperparameter Values

The pre-registered configurations provide reasonable default hyperparameters. A common workflow is to first train a model with one of the presets and then tune some hyperparameters to see if the performance can be further improved.

TextPredictor builds on top of AutoMM, which has a flexible and easy-to-use configuration design. Please refer to :ref:`sec_automm_predictor` on how to customize hyper-parameters.
