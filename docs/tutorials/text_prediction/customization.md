# Text Prediction - Customization
:label:`sec_textprediction_customization`

This tutorial introduces the presets of `TextPredictor` and how to customize hyperparameters.


```{.python .input}
import numpy as np
import warnings
import autogluon as ag
warnings.filterwarnings("ignore")
np.random.seed(123)
```

## Stanford Sentiment Treebank Data

For demonstration, we use the Stanford Sentiment Treebank ([SST](https://nlp.stanford.edu/sentiment/)) dataset.


```{.python .input}
from autogluon.core import TabularDataset
subsample_size = 1000  # subsample for faster demo, you may try specifying larger value
train_data = TabularDataset("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet")
test_data = TabularDataset("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet")
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head(10)
```

## Configure TextPredictor

### Preset Configurations

`TextPredictor` provides several simple preset configurations. Let's take a look at the available presets.


```{.python .input}
from autogluon.text.text_prediction.presets import list_text_presets
list_text_presets()
```

You may be interested in the configuration differences behind the preset strings.


```{.python .input}
list_text_presets(verbose=True)
```

We can find that the main difference between different presets lie in the choices of Huggingface transformer checkpoints. Preset `default` has the same configuration as preset `high_quality`. Designing the presets follows the rule of thumb that larger backbones tend to have better performance but with higher cost.

Let's train a text predictor with preset `medium_quality_faster_train`. 


```{.python .input}
from autogluon.text import TextPredictor
predictor = TextPredictor(eval_metric="acc", label="label")
predictor.fit(
    train_data=train_data,
    presets="medium_quality_faster_train",
    time_limit=60,
)
```

Below we report both `f1` and `acc` metrics for our predictions.


```{.python .input}
predictor.evaluate(test_data, metrics=["f1", "acc"])
```

The pre-registered configurations provide reasonable default hyperparameters. A common workflow is to first train a model with one of the presets and then tune some hyperparameters to see if the performance can be further improved.

### Customize Hyperparameters

Customizing hyperparameters is easy for `TextPredictor`. For example, you may want to try backbones beyond those in the presets. Since `TextPredictor` supports loading Huggingface transformers, you can choose any desired text backbones in the [Hugginface model zoo](https://huggingface.co/models), e.g., `prajjwal1/bert-tiny`.


```{.python .input}
from autogluon.text import TextPredictor
predictor = TextPredictor(eval_metric="acc", label="label")
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
    },
    time_limit=60,
)
```


```{.python .input}
predictor.evaluate(test_data, metrics=["f1", "acc"])
```

`TextPredictor` also supports using the models that are not available in the [Huggingface model zoo](https://huggingface.co/models). To do this, you need to make sure that the models' definition follow Hugginface's AutoModel, AutoConfig, and AutoTokenizer. Let's simulate a local model.


```{.python .input}
import os
from transformers import AutoModel, AutoConfig, AutoTokenizer
model_key = 'prajjwal1/bert-tiny'
local_path = 'custom_local_bert_tiny'

model = AutoModel.from_pretrained(model_key)
config = AutoConfig.from_pretrained(model_key)
tokenizer = AutoTokenizer.from_pretrained(model_key)

model.save_pretrained(local_path)
config.save_pretrained(local_path)
tokenizer.save_pretrained(local_path)
os.listdir(local_path)
```

Now we can use this local model in `TextPredictor`.


```{.python .input}
from autogluon.text import TextPredictor
predictor = TextPredictor(eval_metric="acc", label="label")
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "model.hf_text.checkpoint_name": "custom_local_bert_tiny/",
    },
    time_limit=60,
)
```


```{.python .input}
predictor.evaluate(test_data, metrics=["f1", "acc"])
```

`TextPredictor` builds on top of `AutoMMPredictor`, which has a flexible and easy-to-use configuration design. Please refer to :ref:`sec_automm_predictor` on how to customize more hyperparameters.
