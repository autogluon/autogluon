# AutoMM for Image, Text, and Tabular
:label:`sec_automm_predictor`

AutoMM is a flexible framework that can train neural networks on image, text, numerical, and categorical data. For either single modal or multimodal data, AutoMM supports both classification and regression tasks. This tutorial demonstrates AutoMM's several application scenarios:

- Image Prediction.
- Text Prediction.
- Multimodal Prediction.
    - TIMM + Huggingface Transformers + More
    - CLIP


```{.python .input}
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset

For demonstration, we use the [PetFinder dataset](https://www.kaggle.com/c/petfinder-adoption-prediction). The PetFinder dataset provides information about shelter animals that appear on their adoption profile to predict the animals' adoption rates. The end goal is for rescue shelters to use the predicted adoption rate to identify animals whose profiles could be improved so that they can find a home.

The PetFinder dataset contains image, text, and tabular data. Each animal's adoption profile includes various information, such as pictures of the animal, a text description of the animal, and various tabular features such as age, breed, name, color, and more.

To get started, we first need to download the dataset. Datasets that contain images require more than a CSV file, so the dataset is packaged in a zip file in S3. We will first download it and unzip the contents:


```{.python .input}
download_dir = './ag_petfinder_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip'
```


```{.python .input}
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

Now that the data is download and unzipped, let's take a look at the contents:


```{.python .input}
import os
os.listdir(download_dir)
```

'file.zip' is the original zip file we downloaded, and 'petfinder_processed' is a directory containing the dataset files.


```{.python .input}
dataset_path = download_dir + '/petfinder_processed'
os.listdir(dataset_path)
```

Here we can see the train, test, and dev CSV files, as well as two directories: 'test_images' and 'train_images' which contain the image JPG files.

Note that we will be using the dev data as testing data as dev contains the ground truth labels for showing scores.

Let's take a peek at the first 10 files inside of the 'train_images' directory:


```{.python .input}
os.listdir(dataset_path + '/train_images')[:10]
```

Next, we will load the train and dev CSV files:


```{.python .input}
import pandas as pd

train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)
```


```{.python .input}
train_data.head(3)
```

Looking at the first 3 examples, we can tell that there are various tabular features, a text description ('Description'), and an image path ('Images').

For the PetFinder dataset, we will try to predict the speed of adoption for the animal ('AdoptionSpeed'), grouped into 5 categories, which means that we are dealing with a multi-class classification problem.


```{.python .input}
label_col = 'AdoptionSpeed'
image_col = 'Images'
```

Let's take a look at what a value in the image column looks like:


```{.python .input}
train_data[image_col].iloc[0]
```

Although AutoMM can support training with multiple images per row, we only keep the first image for a quick tutorial.


```{.python .input}
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

train_data[image_col].iloc[0]
```

AutoMM loads images based on the file path provided by the image column.

Here we get the full paths of images.


```{.python .input}
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

train_data[image_col].iloc[0]
```

let's take a look at an example row of data and display the text description and a picture.


```{.python .input}
example_row = train_data.iloc[1]

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

## AutoMM Configuration
It is important to have a flexible and scalable configuration design to handle multiple data modalities, a wide range of model backbones, various training environment setups. Fortunately, AutoMM has a powerful and user-friendly configuration system. You can easily customize its configurations to train diverse models.

First, let's see the available model presets.


```{.python .input}
from autogluon.text.automm.presets import list_model_presets, get_preset
model_presets = list_model_presets()
model_presets
```

You may wonder why AutoMM stays in AutoGluon's text module. AutoMM's current location is temporary, and it will have a new namespace in the next release.

Currently, AutoMM has only one model preset, from which we can construct the predictor's preset.


```{.python .input}
preset = get_preset(model_presets[0])
preset
```

AutoMM's configurations consist of four parts: `model`, `data`, `optimization`, and `environment`. You can convert the preset to configurations to see the details.


```{.python .input}
from omegaconf import OmegaConf
from autogluon.text.automm.utils import get_config
config = get_config(preset)
OmegaConf.to_container(config)
```

`model` config provides four model types: MLP for categorical data (categorical_mlp), MLP for numerical data (numerical_mlp), [huggingface transformers](https://github.com/huggingface/transformers) for text data (hf_text), [timm](https://github.com/rwightman/pytorch-image-models) for image data (timm_image), clip for image+text data, and a MLP to fuse any combinations of categorical_mlp, numerical_mlp, hf_text, and timm_image (fusion_mlp). We can specify the model combinations by setting `model.names`. Moreover, we can use `model.hf_text.checkpoint_name` and `model.timm_image.checkpoint_name` to customize huggingface and timm backbones.

`data` config defines some model-agnostic rules in preprocessing data. Note that AutoMM converts categorical data into text by default.

`optimization` config has hyper-parameters for model training. AutoMM uses layer-wise learning rate decay, which decreases the learning rate gradually from the output to the input end of one model.

`env` contains the environment/machine related configurations. For example, the optimal values of `per_gpu_batch_size` and `per_gpu_batch_size_evaluation` are closely related to the GPU memory size.

You can flexibly customize any hyper-parameter in `config` via the `hyperparameters` argument of `.fit()`. To access one hyper-parameter in `config`, you need to traverse from top-level keys to bottom-level keys and join them together with `.` For example, if you want to change the per gpu batch size to 16, you can set `hyperparameters={"env.per_gpu_batch_size": 16}`.


Next, we will introduce how to customize the default config to train different models.

## Image Prediction

Let's first train an image classifier on the PetFinder dataset. AutoMM supports loading diverse [timm](https://github.com/rwightman/pytorch-image-models) backbones. We can choose one image backbone, e.g., [swin_tiny_patch4_window7_224](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py#L69), by using hyper-parameter `model.timm_image.checkpoint_name`.


```{.python .input}
from autogluon.text.automm import AutoMMPredictor
predictor = AutoMMPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    config=config,
    hyperparameters={
        "model.names": ["timm_image"],
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
    },
    time_limit=60, # seconds
)
```

During `.fit()`, `train_data` is automatically divided into training and validation sets. AutoMM uses the validation score to select the top k (default 3) model checkpoints and average them as the final model. When calling `.fit()` you can also set `tuning_data=validation_data` to use your specified `validation_data`.

Here AutoMM uses only image data and automatically ignores other data since `model.names` only include `timm_image`.

Then we can evaluate the trained model on test data. AutoMM supports evaluation with multiple metrics.


```{.python .input}
scores = predictor.evaluate(test_data, metrics=["accuracy", "quadratic_kappa"])
print(scores)
```

Here we use a tiny backbone and a tight time budget for a quick demonstration. To perform well on your tasks, you may choose a larger backbone and do not set a time limit.

Given data without ground truth labels, AutoMM can make predictions.


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

Extract embeddings can be easily done via `.extract_embedding()`


```{.python .input}
embeddings = predictor.extract_embedding(test_data.drop(columns=label_col))
embeddings[0]
```

Moreover, AutoMM provides the APIs for saving and loading predictors.


```{.python .input}
predictor.save('my_saved_dir')
loaded_predictor = AutoMMPredictor.load('my_saved_dir')
scores2 = loaded_predictor.evaluate(test_data, metrics=["accuracy", "quadratic_kappa"])
print(scores2)
```

We can observe that `scores2` are consistent with `scores`.

## Text Prediction

Training text models is also convenient through AutoMM, which supports finetuning from [huggingface transformers](https://github.com/huggingface/transformers) backbones. You can search in [huggingface model zoo](https://huggingface.co/models) to find suitable model checkpoint names, e.g, [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny).


```{.python .input}
from autogluon.text.automm import AutoMMPredictor
predictor = AutoMMPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    config=config,
    hyperparameters={
        "model.names": ["hf_text"],
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
    },
    time_limit=60, # seconds
)
```

Although `train_data` has different data modalities, `.fit()` only uses the text data since we have set `model.names` with only `hf_text`.

Like the above image prediction, we can either evaluate on test data, 


```{.python .input}
scores = predictor.evaluate(test_data, metrics=["accuracy", "quadratic_kappa"])
print(scores)
```

predict on unlabeled data,


```{.python .input}
predictions = predictor.predict(test_data.drop(columns=label_col))
predictions[:5]
```

get class probabilities,


```{.python .input}
probas = predictor.predict_proba(test_data.drop(columns=label_col))
probas[:5]
```

or extract embeddings.


```{.python .input}
embeddings = predictor.extract_embedding(test_data.drop(columns=label_col))
embeddings[0]
```

Likewise, we can save the predictor, load it, and get the same evaluatiion scores.


```{.python .input}
predictor.save('my_saved_dir')
loaded_predictor = AutoMMPredictor.load('my_saved_dir')
scores2 = loaded_predictor.evaluate(test_data, metrics=["accuracy", "quadratic_kappa"])
print(scores2)
```

This quick demo probably results in low scores due to the tiny backbone, subsampled data, and time limit. You may try a larger backbone and avoid setting the time limit in practice.

## Multimodal Prediction
One significant merit of AutoMM is its scalability to multiple modalities, including image, text, numerical, and categorical data. Given one multimodal dataframe, AutoMM can automatically detect the modality of each column and then select the suitable backbones for different modalities. If more than one individual backbones are available, it would add a MLP fusion module on the top to fuse their output features to make the final prediction. Thanks to the flexible design, it's easy to apply AutoMM to any image, text, numerical, and categorical data combination. AutoMM can also infer the problem type (classification or regression) by analyzing the label column.

### TIMM + Huggingface Transformers + More
AutoMM maximizes the choices of image and text backbones by supporting [timm](https://github.com/rwightman/pytorch-image-models) and [huggingface transformers](https://github.com/huggingface/transformers). AutoMM uses MLP for numerical data but converts categorical data to text by default. 

Let's apply AutoMM to all the data modalities in the PetFinder dataset. We also use the small image and text backbones as above to speed up training.


```{.python .input}
from autogluon.text.automm import AutoMMPredictor
predictor = AutoMMPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    config=config,
    hyperparameters={
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
    },
    time_limit=180, # seconds
)
```

Then we can evaluate the multimodal predictor's performance.


```{.python .input}
scores = predictor.evaluate(test_data, metrics=["accuracy", "quadratic_kappa"])
print(scores)
```

Moreover, AutoMM's APIs `.predict()`, `predict_proba.()`, `extract_embedding.()`, `.save()`, and `.load()` can also work with the multimodal predictor. Please refer to Section Image Prediction for examples.

### CLIP
For vision language tasks, we can also finetune the pretrained vision language models, such as CLIP.


```{.python .input}
from autogluon.text.automm import AutoMMPredictor
predictor = AutoMMPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    config=config,
    hyperparameters={
        "model.names": ["clip"],
    },
    time_limit=120, # seconds
)
```


```{.python .input}
scores = predictor.evaluate(test_data, metrics=["accuracy", "quadratic_kappa"])
print(scores)
```

Please refer to examples in Section Image Prediction to use other APIs', such as `.predict()`, `predict_proba.()`, `extract_embedding.()`, `.save()`, and `.load()`.
