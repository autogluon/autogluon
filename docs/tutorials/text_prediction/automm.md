# AutoMMPredictor for Image, Text, and Tabular
:label:`sec_automm_predictor`

Are you tired of switching codebases or hacking code for different data modalities (image, text, numerical, and categorical data) and tasks (classification, regression, and more)? `AutoMMPredictor` provides a one-stop shop for multimodal/unimodal deep learning models. This tutorial demonstrates several application scenarios.

- Multimodal Prediction
    - CLIP
    - TIMM + Huggingface Transformers + More
- Image Prediction
- Text Prediction
- Configuration Customization
- APIs


```{.python .input}
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset

For demonstration, we use the [PetFinder dataset](https://www.kaggle.com/c/petfinder-adoption-prediction). The PetFinder dataset provides information about shelter animals that appear on their adoption profile to predict the animals' adoption rates, grouped into five categories, hence a multi-class classification problem.

To get started, let's download and prepare the dataset.


```{.python .input}
download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

Next, we will load the CSV files.


```{.python .input}
import pandas as pd
dataset_path = download_dir + '/petfinder_processed'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)
label_col = 'AdoptionSpeed'
```

We need to expand the image paths to load them in training.


```{.python .input}
image_col = 'Images'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0]) # Use the first image for a quick tutorial
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])


def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

train_data[image_col].iloc[0]
```

Each animal's adoption profile includes pictures, a text description, and various tabular features such as age, breed, name, color, and more. Let's look at an example row of data and display the text description and a picture.


```{.python .input}
example_row = train_data.iloc[47]

example_row
```


```{.python .input}
example_row['Description']
```


```{.python .input}
example_image = example_row['Images']

from IPython.display import Image, display
pil_img = Image(filename=example_image)
display(pil_img)
```

For the demo purpose, we will sample 500 and 100 rows for training and testing, respectively.


```{.python .input}
train_data = train_data.sample(500, random_state=0)
test_data = test_data.sample(100, random_state=0)
```

## Multimodal Prediction
### CLIP
`AutoMMPredictor` allows for finetuning the pre-trained vision language models, such as [CLIP](https://huggingface.co/openai/clip-vit-base-patch32).


```{.python .input}
from autogluon.text.automm import AutoMMPredictor
predictor = AutoMMPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "model.names": ["clip"],
        "env.num_gpus": 1,
    },
    time_limit=120, # seconds
)
```


```{.python .input}
scores = predictor.evaluate(test_data, metrics=["accuracy"])
scores
```

In this example, `AutoMMPredictor` finetunes CLIP with the image, text, and categorical (converted to text) data.

### TIMM + Huggingface Transformers + More
In addtion to CLIP, `AutoMMPredictor` can simultaneously finetune various [timm](https://github.com/rwightman/pytorch-image-models) backbones and [huggingface transformers](https://github.com/huggingface/transformers). Moreover, `AutoMMPredictor` uses MLP for numerical data but converts categorical data to text by default. 

Let's use `AutoMMPredictor` to train a late fusion model including [CLIP](https://huggingface.co/openai/clip-vit-base-patch32), [swin_small_patch4_window7_224](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py#L65), [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator), a numerical MLP, and a fusion MLP. 


```{.python .input}
from autogluon.text.automm import AutoMMPredictor
predictor = AutoMMPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "model.names": ["clip", "timm_image", "hf_text", "numerical_mlp", "fusion_mlp"],
        "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
        "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
        "env.num_gpus": 1,
    },
    time_limit=120, # seconds
)
```


```{.python .input}
scores = predictor.evaluate(test_data, metrics=["accuracy"])
scores
```

## Image Prediction

If you want to use only image data or your tasks only have image data, `AutoMMPredictor` can help you finetune a wide range of [timm](https://github.com/rwightman/pytorch-image-models) backbones, such as [swin_small_patch4_window7_224](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py#L65).


```{.python .input}
from autogluon.text.automm import AutoMMPredictor
predictor = AutoMMPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "model.names": ["timm_image"],
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
    },
    time_limit=60, # seconds
)
```

Here `AutoMMPredictor` uses only image data since `model.names` only include `timm_image`.


```{.python .input}
scores = predictor.evaluate(test_data, metrics=["accuracy"])
scores
```

## Text Prediction
Similarly, you may be interested in only finetuning the text backbones from [huggingface transformers](https://github.com/huggingface/transformers), such as [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator).


```{.python .input}
from autogluon.text.automm import AutoMMPredictor
predictor = AutoMMPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "model.names": ["hf_text"],
        "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
        "env.num_gpus": 1,
    },
    time_limit=60, # seconds
)
```

With only `hf_text` in `model.names`, `AutoMMPredictor` automatically uses only text and categorical (converted to text) data.


```{.python .input}
scores = predictor.evaluate(test_data, metrics=["accuracy"])
scores
```

## Configuration Customization
The above examples have shown the flexibility of `AutoMMPredictor`. You may want to know how to customize configurations for your tasks. Fortunately, `AutoMMPredictor` has a user-friendly configuration design.

First, let's see the available model presets.


```{.python .input}
from autogluon.text.automm.presets import list_model_presets, get_preset
model_presets = list_model_presets()
model_presets
```

Currently, `AutoMMPredictor` has only one model preset, from which we can construct the predictor's preset.


```{.python .input}
preset = get_preset(model_presets[0])
preset
```

`AutoMMPredictor` configurations consist of four parts: `model`, `data`, `optimization`, and `environment`. You can convert the preset to configurations to see the details.


```{.python .input}
from omegaconf import OmegaConf
from autogluon.text.automm.utils import get_config
config = get_config(preset)
print(OmegaConf.to_yaml(config))
```

The `model` config provides four model types: MLP for categorical data (categorical_mlp), MLP for numerical data (numerical_mlp), [huggingface transformers](https://github.com/huggingface/transformers) for text data (hf_text), [timm](https://github.com/rwightman/pytorch-image-models) for image data (timm_image), clip for image+text data, and a MLP to fuse any combinations of categorical_mlp, numerical_mlp, hf_text, and timm_image (fusion_mlp). We can specify the model combinations by setting `model.names`. Moreover, we can use `model.hf_text.checkpoint_name` and `model.timm_image.checkpoint_name` to customize huggingface and timm backbones.

The `data` config defines some model-agnostic rules in preprocessing data. Note that `AutoMMPredictor` converts categorical data into text by default.

The `optimization` config has hyper-parameters for model training. `AutoMMPredictor` uses layer-wise learning rate decay, which decreases the learning rate gradually from the output to the input end of one model.

The `env` config contains the environment/machine related hyper-parameters. For example, the optimal values of `per_gpu_batch_size` and `per_gpu_batch_size_evaluation` are closely related to the GPU memory size.

You can flexibly customize any hyper-parameter in `config` via the `hyperparameters` argument of `.fit()`. To access one hyper-parameter in `config`, you need to traverse from top-level keys to bottom-level keys and join them together with `.` For example, if you want to change the per GPU batch size to 16, you can set `hyperparameters={"env.per_gpu_batch_size": 16}`.

## APIs
Besides `.fit()` and `.evaluate()`, `AutoMMPredictor` also provides other useful APIs, similar to those in `TextPredictor` and `TabularPredictor`. You may refer to more details in :ref:`sec_textprediction_beginner`.

Given data without ground truth labels, `AutoMMPredictor` can make predictions.


```{.python .input}
predictions = predictor.predict(test_data.drop(columns=label_col))
predictions[:5]
```

For classification tasks, we can get the probabilities of all classes.


```{.python .input}
probas = predictor.predict_proba(test_data.drop(columns=label_col))
probas[:5]
```

Note that calling `.predict_proba` on one regression task will throw an exception.

Extract embeddings can be easily done via `.extract_embedding()`.


```{.python .input}
embeddings = predictor.extract_embedding(test_data.drop(columns=label_col))
embeddings.shape
```

It is also convenient to save and load a predictor.


```{.python .input}
predictor.save('my_saved_dir')
loaded_predictor = AutoMMPredictor.load('my_saved_dir')
scores2 = loaded_predictor.evaluate(test_data, metrics=["accuracy"])
scores2
```
