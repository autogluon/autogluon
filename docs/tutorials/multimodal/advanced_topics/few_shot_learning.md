# Few Shot Learning with FusionSVM Model
:label:`sec_automm_few_shot_learning`

In this tutorial we introduce a simple but effective way for few shot classification problems. 
We present the FusionSVM model which leverages the high-quality features from foundational models and use a simple SVM for few shot classification task.
Specifically, we extract sample features with pretrained models, and use the features for SVM learning.
We show the effectiveness of this FusionSVMModel on a text classification dataset and a vision classification dataset. 

## Text Classification on MLDoc dataset
### Load Dataset
We prepare all datasets in the format of `pd.DataFrame` as in many of our tutorials have done. 
For this tutorial, we'll use a small `MLDoc` dataset for demonstration. 
The dataset is a text classification dataset, which contains 4 classes and we downsampled the training data to 10 samples per class, a.k.a 10 shots.
For more details regarding `MLDoc` please see this [link](https://github.com/facebookresearch/MLDoc).

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

### Create the `FewShotSVMPredictor`
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

### Train the model
Now we train the model with the `train_df`.
```{.python .input}
predictor.fit(train_df)
```

### Run evaluation
```{.python .input}
result = predictor.evaluate(test_df, metrics=["acc", "macro_f1"])
print(result)
```

### Comparing to the normal `MultiModalPredictor`
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
    problem_type="classification",
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

### Load a pretrained model
The `FewShotSVMPredictor` automatically saves the model and artifacts to disk during training. 
You can specify the path to save by setting the `path=<your_desired_save_path>` when initializing the predictor.
You can also load a pretrained `FewShotSVMPredictor` and perform downstream tasks by the following code:

```{.python .input}
predictor2 = FewShotSVMPredictor.load(model_path)
result2 = predictor2.evaluate(test_df, metrics=["acc", "macro_f1"])
print(result2)
```

## Image Classification on Stanford Cars
### Load Dataset
We also provide an example of using `FewShotSVMPredictor` on a few-shot image classification task. 
We use the Stanford Cars dataset for demonstration and downsampled the training set to have 8 samples per class.
The Stanford Cars is an image classification dataset and contains 196 classes.
For more information regarding the dataset, please see [here](http://ai.stanford.edu/~jkrause/cars/car_dataset.html).

```{.python .input}
import pandas as pd
import os

from autogluon.core.utils.loaders import load_zip, load_s3

download_dir = "./ag_automm_tutorial_fs_cls/stanfordcars/"
zip_file = "https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/stanfordcars.zip"
train_csv = "https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/train_8shot.csv"
test_csv = "https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/test.csv"

load_zip.unzip(zip_file, unzip_dir=download_dir)
dataset_path = os.path.join(download_dir)

!wget https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/train_8shot.csv -O ./ag_automm_tutorial_fs_cls/stanfordcars/train.csv
!wget https://automl-mm-bench.s3.amazonaws.com/vision_datasets/stanfordcars/test.csv -O ./ag_automm_tutorial_fs_cls/stanfordcars/test.csv

train_df_raw = pd.read_csv(os.path.join(download_dir, "train.csv"))
train_df = train_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
train_df["ImageID"] = download_dir + train_df["ImageID"].astype(str)


test_df_raw = pd.read_csv(os.path.join(download_dir, "test.csv"))
test_df = test_df_raw.drop(
        columns=[
            "Source",
            "Confidence",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsOccluded",
            "IsTruncated",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    )
test_df["ImageID"] = download_dir + test_df["ImageID"].astype(str)

print(os.path.exists(train_df.iloc[0]["ImageID"]))
print(train_df)
print(os.path.exists(test_df.iloc[0]["ImageID"]))
print(test_df)
```

### Create the `FewShotSVMPredictor`
In order to run FusionSVM model, we first initialize a `FewShotSVMPredictor` with the following parameters.
```{.python .input}
from autogluon.multimodal.utils.few_shot_learning import FewShotSVMPredictor
hyperparameters = {
    "model.names": ["clip"],
    "model.clip.max_text_len": 0,
    "env.num_workers": 2,
    "model.clip.checkpoint_name": "openai/clip-vit-large-patch14-336",
    "env.eval_batch_size_ratio": 1,
}

import uuid
model_path = f"./tmp/{uuid.uuid4().hex}-automm_stanfordcars-8shot-en"
predictor = FewShotSVMPredictor(
    label="LabelName",  # column name of the label
    hyperparameters=hyperparameters,
    eval_metric="acc",
    path=model_path  # path to save model and artifacts
)
```

### Train the model
Now we train the model with the `train_df`.
```{.python .input}
predictor.fit(train_df)
```

### Run evaluation
```{.python .input}
result = predictor.evaluate(test_df, metrics=["acc", "macro_f1"])
print(result)
```

### Comparing to the normal `MultiModalPredictor`
```{.python .input} 
from autogluon.multimodal import MultiModalPredictor
import numpy as np
from sklearn.metrics import f1_score


hyperparameters = {
    "model.names": ["timm_image"],
    "model.timm_image.checkpoint_name": "swin_base_patch4_window7_224",
    "env.per_gpu_batch_size": 8,
    "optimization.max_epochs": 10,
    "optimization.learning_rate": 1.0e-3,
    "optimization.optim_type": "adamw",
    "optimization.weight_decay": 1.0e-3,
}

automm_predictor = MultiModalPredictor(
    label="LabelName",  # column name of the label
    hyperparameters=hyperparameters,
    problem_type="classification",
    eval_metric="acc",
)
automm_predictor.fit(
    train_data=train_df,
)

results, preds = automm_predictor.evaluate(test_df, return_pred=True)
test_labels = np.array(test_df["LabelName"])
macro_f1 = f1_score(test_labels, preds, average="macro")
results["macro_f1"] = macro_f1

print(results)
```

As you can see that the `FewShotSVMPredictor` performs much better than the normal `MultiModalPredictor` in image classification as well.


### Citation
```
@InProceedings{SCHWENK18.658,
  author = {Holger Schwenk and Xian Li},
  title = {A Corpus for Multilingual Document Classification in Eight Languages},
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year = {2018},
  month = {may},
  date = {7-12},
  location = {Miyazaki, Japan},
  editor = {Nicoletta Calzolari (Conference chair) and Khalid Choukri and Christopher Cieri and Thierry Declerck and Sara Goggi and Koiti Hasida and Hitoshi Isahara and Bente Maegaard and Joseph Mariani and Hélène Mazo and Asuncion Moreno and Jan Odijk and Stelios Piperidis and Takenobu Tokunaga},
  publisher = {European Language Resources Association (ELRA)},
  address = {Paris, France},
  isbn = {979-10-95546-00-9},
  language = {english}
  }
  
@inproceedings{KrauseStarkDengFei-Fei_3DRR2013,
  title = {3D Object Representations for Fine-Grained Categorization},
  booktitle = {4th International IEEE Workshop on  3D Representation and Recognition (3dRR-13)},
  year = {2013},
  address = {Sydney, Australia},
  author = {Jonathan Krause and Michael Stark and Jia Deng and Li Fei-Fei}
}
```