# Few Shot Learning with FusionSVM Model
:label:`sec_automm_few_shot_learning`

In this tutorial we introduce a simple but effective way for few shot classification problems. 
We present the FusionSVM model which leverages the high-quality features from foundational models and use a simple SVM for few shot classification task.
Specifically, we extract sample features with pretrained models, and use the features for SVM learning.

## Load Dataset
We prepare all datasets in the format of `pd.DataFrame` as in many of our tutorials have done. 
For this tutorial, we'll use a small `shopee` dataset for demonstration.

```{.python .input}
import pandas as pd
import os

from autogluon.core.utils.loaders import load_zip

download_dir = "./ag_automm_tutorial_fs_cls"
zip_file = "https://automl-mm-bench.s3.amazonaws.com/nlp_datasets/MLDoc-10shot-en.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)
dataset_path = os.path.join(download_dir)
train_df = pd.read_csv(f"{dataset_path}/train.csv", names=["label", "text"])
test_df = pd.read_csv(f"{dataset_path}/test.csv", names=["label", "text"])
print(train_df)
print(test_df)
```

## Create the `FewShotSVMPredictor`
In order to run FusionSVM model, we first initialize a `FewShotSVMPredictor` with the following parameters.
```{.python .input}
from autogluon.multimodal.utils.few_shot_learning import FewShotSVMPredictor
hyperparameters = {
    "model.hf_text.checkpoint_name": "sentence-transformers/all-mpnet-base-v2",
    "model.hf_text.pooling_mode": "mean",
    "env.per_gpu_batch_size": 32,
    "env.eval_batch_size_ratio": 4,
}

import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-automm_mldoc-10shot-en"
predictor = FewShotSVMPredictor(
    label="label",  # column name of the label
    hyperparameters=hyperparameters,
    eval_metric="acc",
    path=model_path  # path to save model and artifacts
)
```

## Train the model
Now we train the model with the `train_df`.
```{.python .input}
predictor.fit(train_df)
```

## Run evaluation
```{.python .input}
result = predictor.evaluate(test_df, metrics=["acc", "macro_f1"])
print(result)
```

## Comparing to the normal `MultiModalPredictor`
```{.python .input} 
from autogluon.multimodal import MultiModalPredictor
import numpy as np
from sklearn.metrics import f1_score

hyperparameters = {
    "model.hf_text.checkpoint_name": "sentence-transformers/all-mpnet-base-v2",
    "model.hf_text.pooling_mode": "mean",
    "env.per_gpu_batch_size": 32,
    "env.eval_batch_size_ratio": 4,
}

automm_predictor = MultiModalPredictor(
    label="label",
    problem_type="Classification",
    eval_metric="acc"
)

automm_predictor.fit(
    train_data=train_df,
    presets="multilingual",
    hyperparameters=hyperparameters,
)

results, preds = automm_predictor.evaluate(test_df, return_pred=True)
test_labels = np.array(test_df["label"])
macro_f1 = f1_score(test_labels, preds, average="macro")
results["macro_f1"] = macro_f1

print(results)
```

As you can see that the `FewShotSVMPredictor` performs much better than the normal `MultiModalPredictor`. 

## Load a pretrained model
The `FewShotSVMPredictor` automatically saves the model and artifacts to disk during training. 
You can specify the path to save by setting the `path=<your_desired_save_path>` when initializing the predictor.
You can also load a pretrained `FewShotSVMPredictor` and perform downstream tasks by the following code:

```{.python .input}
predictor2 = FewShotSVMPredictor.load(model_path)
result2 = predictor2.evaluate(test_df, metrics=["acc", "macro_f1"])
print(result2)
```