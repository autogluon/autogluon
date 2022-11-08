# AutoMM for Multimodal - Quick Start
:label:`sec_automm_multimodal_beginner`

AutoMM is a deep learning "model zoo" of model zoos. It can automatically build deep learning models that are suitable for multimodal datasets. You will only need to convert the data into the multimodal dataframe format
and AutoMM can predict the values of one column conditioned on the features from the other columns including images, text, and tabular data.


```{.python .input}
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset

For demonstration, we use a simplified and subsampled version of [PetFinder dataset](https://www.kaggle.com/c/petfinder-adoption-prediction). The task is to predict the animals' adoption rates based on their adoption profile information. In this simplified version, the adoption speed is grouped into two categories: 0 (slow) and 1 (fast).

To get started, let's download and prepare the dataset.


```{.python .input}
download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

Next, we will load the CSV files.


```{.python .input}
import pandas as pd
dataset_path = download_dir + '/petfinder_for_tutorial'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
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
example_row = train_data.iloc[0]

example_row
```


```{.python .input}
example_row['Description']
```


```{.python .input}
example_image = example_row[image_col]

from IPython.display import Image, display
pil_img = Image(filename=example_image)
display(pil_img)
```

## Training
Now let's fit the predictor with the training data. Here we set a tight time budget for a quick demo.

```{.python .input}
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label=label_col)
predictor.fit(
    train_data=train_data,
    time_limit=120, # seconds
)
```
Under the hood, AutoMM automatically infers the problem type (classification or regression), detects the data modalities, selects the related models from the multimodal model pools, and trains the selected models. If multiple backbones are available, AutoMM appends a late-fusion model (MLP or transformer) on top of them.


## Evaluation
Then we can evaluate the predictor on the test data.
```{.python .input}
scores = predictor.evaluate(test_data, metrics=["roc_auc"])
scores
```


## Prediction
Given a multimodal dataframe without the label column, we can predict the labels.


```{.python .input}
predictions = predictor.predict(test_data.drop(columns=label_col))
predictions[:5]
```

For classification tasks, we can get the probabilities of all classes.


```{.python .input}
probas = predictor.predict_proba(test_data.drop(columns=label_col))
probas[:5]
```

Note that calling `.predict_proba()` on one regression task will throw an exception.


## Extract Embeddings

Extracting embeddings can also be useful in many cases, where we want to convert each sample (per row in the dataframe) into an embedding vector.

```{.python .input}
embeddings = predictor.extract_embedding(test_data.drop(columns=label_col))
embeddings.shape
```


## Save and Load
It is also convenient to save a predictor and re-load it.


```{.python .input}
predictor.save('my_saved_dir')
loaded_predictor = MultiModalPredictor.load('my_saved_dir')
scores2 = loaded_predictor.evaluate(test_data, metrics=["roc_auc"])
scores2
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
