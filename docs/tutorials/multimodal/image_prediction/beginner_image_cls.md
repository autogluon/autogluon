# AutoMM for Image Classification - Quick Start
:label:`sec_automm_imageclassification_beginner`

In this quick start, we'll use the task of image classification to illustrate how to use **MultiModalPredictor**. Once the data is prepared in [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) format, a single call to `MultiModalPredictor.fit()` will take care of the model training for you.


## Create Image Dataset

For demonstration purposes, we use a subset of the [Shopee-IET dataset](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) from Kaggle.
Each image in this data depicts a clothing item and the corresponding label specifies its clothing category.
Our subset of the data contains the following possible labels: `BabyPants`, `BabyShirt`, `womencasualshoes`, `womenchiffontop`.

We can load a dataset by downloading a url data automatically:

```{.python .input}
import warnings
warnings.filterwarnings('ignore')
from autogluon.vision import ImageDataset
train_dataset, _, test_dataset = ImageDataset.from_folders("https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip")
print(train_dataset)
```

We can see there are 800 rows and 2 columns in this training dataframe. The 2 columns are **image** and **label**, and each row represents a different training sample.


## Use AutoMM to Fit Models

Now, we fit a classifier using AutoMM as follows:

```{.python .input}
from autogluon.multimodal import MultiModalPredictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-automm_imgcls"
predictor = MultiModalPredictor(label="label", path=model_path)
predictor.fit(
    train_data=train_dataset,
    time_limit=60, # seconds
) # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model
```

**label** is the name of the column that contains the target variable to predict, e.g., it is "label" in our example. **path** indicates the directory where models and intermediate outputs should be saved. We set the training time limit to 30 seconds for demonstration purpose, but you can control the training time by setting configurations. To customize AutoMM, please refer to :ref:`sec_automm_customization`.


## Evaluate on Test Dataset

You can evaluate the classifier on the test dataset to see how it performs, the test top-1 accuracy is:

```{.python .input}
scores = predictor.evaluate(test_dataset, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores["accuracy"])
```


## Predict on a New Image

Given an example image, let's visualize it first,

```{.python .input}
image_path = test_dataset.iloc[0]['image']
from IPython.display import Image, display
pil_img = Image(filename=image_path)
display(pil_img)
```

We can easily use the final model to `predict` the label,

```{.python .input}
predictions = predictor.predict({'image': [image_path]})
print(predictions)
```

If probabilities of all categories are needed, you can call `predict_proba`:

```{.python .input}
proba = predictor.predict_proba({'image': [image_path]})
print(proba)
```


## Extract Embeddings

Extracting representation from the whole image learned by a model is also very useful. We provide `extract_embedding` function to allow predictor to return the N-dimensional image feature where `N` depends on the model(usually a 512 to 2048 length vector)

```{.python .input}
feature = predictor.extract_embedding({'image': [image_path]})
print(feature[0].shape)
```


## Save and Load

The trained predictor is automatically saved at the end of `fit()`, and you can easily reload it.

:::warning

`MultiModalPredictor.load()` used `pickle` module implicitly, which is known to be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Never load data that could have come from an untrusted source, or that could have been tampered with. **Only load data you trust.**

:::

```{.python .input}
loaded_predictor = MultiModalPredictor.load(model_path)
load_proba = loaded_predictor.predict_proba({'image': [image_path]})
print(load_proba)
```

We can see the predicted class probabilities are still the same as above, which means same model!

## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
