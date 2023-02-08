# Multimodal Data Tables: Tabular, Text, and Image

:label:`sec_tabularprediction_multimodal`

**Tip**: Prior to reading this tutorial, it is recommended to have a basic understanding of the TabularPredictor API covered in :ref:`sec_tabularquick`.

In this tutorial, we will train a multi-modal ensemble using data that contains image, text, and tabular features.

Note: A GPU is required for this tutorial in order to train the image and text models. Additionally, GPU installations are required for MXNet and Torch with appropriate CUDA versions.

## The PetFinder Dataset

We will be using the [PetFinder dataset](https://www.kaggle.com/c/petfinder-adoption-prediction). The PetFinder dataset provides information about shelter animals that appear on their adoption profile with the goal to predict the adoption rate of the animal. The end goal is for rescue shelters to use the predicted adoption rate to identify animals whose profiles could be improved so that they can find a home.

Each animal's adoption profile contains a variety of information, such as pictures of the animal, a text description of the animal, and various tabular features such as age, breed, name, color, and more.

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

Note: We will be using the dev data as testing data as dev contains the ground truth labels for showing scores via `predictor.leaderboard`.

Let's take a peek at the first 10 files inside of the 'train_images' directory:


```{.python .input}
os.listdir(dataset_path + '/train_images')[:10]
```

As expected, these are the images we will be training with alongside the other features.

Next, we will load the train and dev CSV files:


```{.python .input}
import pandas as pd

train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)
```


```{.python .input}
train_data.head(3)
```

Looking at the first 3 examples, we can tell that there is a variety of tabular features, a text description ('Description'), and an image path ('Images').

For the PetFinder dataset, we will try to predict the speed of adoption for the animal ('AdoptionSpeed'), grouped into 5 categories. This means that we are dealing with a multi-class classification problem.


```{.python .input}
label = 'AdoptionSpeed'
image_col = 'Images'
```

## Preparing the image column

Let's take a look at what a value in the image column looks like:


```{.python .input}
train_data[image_col].iloc[0]
```

Currently, AutoGluon only supports one image per row. Since the PetFinder dataset contains one or more images per row, we first need to preprocess the image column to only contain the first image of each row.


```{.python .input}
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

train_data[image_col].iloc[0]
```

AutoGluon loads images based on the file path provided by the image column.

Here we update the path to point to the correct location on disk:


```{.python .input}
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

train_data[image_col].iloc[0]
```


```{.python .input}
train_data.head(3)
```

## Analyzing an example row

Now that we have preprocessed the image column, let's take a look at an example row of data and display the text description and the picture.


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

The PetFinder dataset is fairly large. For the purposes of the tutorial, we will sample 300 rows for training.

Training on large multi-modal datasets can be very computationally intensive, especially if using the `best_quality` preset in AutoGluon. When prototyping, it is recommended to sample your data to get an idea of which models are worth training, then gradually train with larger amounts of data and longer time limits as you would with any other machine learning algorithm.


```{.python .input}
train_data = train_data.sample(300, random_state=0)
```

## Constructing the FeatureMetadata

Next, let's see what AutoGluon infers the feature types to be by constructing a FeatureMetadata object from the training data:


```{.python .input}
from autogluon.tabular import FeatureMetadata
feature_metadata = FeatureMetadata.from_df(train_data)

print(feature_metadata)
```

Notice that FeatureMetadata automatically identified the column 'Description' as text, so we don't need to manually specify that it is text.

In order to leverage images, we need to tell AutoGluon which column contains the image path. We can do this by specifying a FeatureMetadata object and adding the 'image_path' special type to the image column. We later pass this custom FeatureMetadata to TabularPredictor.fit.


```{.python .input}
feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})

print(feature_metadata)
```

## Specifying the hyperparameters

Next, we need to specify the models we want to train with. This is done via the `hyperparameters` argument to TabularPredictor.fit.

AutoGluon has a predefined config that works well for multimodal datasets called 'multimodal'. We can access it via:


```{.python .input}
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
hyperparameters = get_hyperparameter_config('multimodal')

hyperparameters
```

This hyperparameter config will train a variety of Tabular models as well as finetune an Electra BERT text model, and a ResNet image model.

## Fitting with TabularPredictor

Now we will train a TabularPredictor on the dataset, using the feature metadata and hyperparameters we defined prior. This TabularPredictor will leverage tabular, text, and image features all at once.


```{.python .input}
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label=label).fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    feature_metadata=feature_metadata,
    time_limit=600,
)
```

After the predictor is fit, we can take a look at the leaderboard and see the performance of the various models:


```{.python .input}
leaderboard = predictor.leaderboard(test_data)
```

That's all it takes to train with image, text, and tabular data (at the same time) using AutoGluon!

For an in-depth tutorial on text + tabular multimodal functionality, refer to :ref:`sec_tabularprediction_text_multimodal`.

For more tutorials, refer to :ref:`sec_tabularquick` and :ref:`sec_tabularadvanced`.
