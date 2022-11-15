# Image-to-Image Semantic Matching with AutoMM 
:label:`image2image_matching`

Computing the similarity between two images is a common task in computer vision, with several practical applications such as detecting same or different product, etc. In general, image similarity models will take two images as input and transform them into vectors, and then similarity scores calculated using cosine similarity, dot product, or Euclidean distances are used to measure how alike or different of the two images. 

```{.python .input}
import os
import pandas as pd
import warnings
from IPython.display import Image, display
warnings.filterwarnings('ignore')
```

## Prepare your Data
In this tutorial, we will demonstrate how to use AutoMM for image-to-image semantic matching with the simplified Stanford Online Products dataset ([SOP](https://cvgl.stanford.edu/projects/lifted_struct/)). 

Stanford Online Products dataset is introduced for metric learning. There are 12 categories of products in this dataset: bicycle, cabinet, chair, coffee maker, fan, kettle, lamp, mug, sofa, stapler, table and toaster. Each category has some products, and each product has several images captured from different views. Here, we consider different views of the same product as positive pairs (labeled as 1) and images from different products as negative pairs (labeled as 0). 

The following code downloads the dataset and unzip the images and annotation files.


```{.python .input}
download_dir = './ag_automm_tutorial_img2img'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/Stanford_Online_Products.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

Then we can load the annotations into dataframes.
```{.python .input}
dataset_path = os.path.join(download_dir, 'Stanford_Online_Products')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col_1 = "Image1"
image_col_2 = "Image2"
label_col = "Label"
match_label = 1
```
Here you need to specify the `match_label`, the label class representing that a pair semantically match. In this demo dataset, we use 1 since we assigned 1 to image pairs from the same product. You may consider your task context to specify `match_label`.

Next, we expand the image paths since the original paths are relative.
```{.python .input}
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

for image_col in [image_col_1, image_col_2]:
    train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

The annotations are only image path pairs and their binary labels (1 and 0 mean the image pair matching or not, respectively).
```{.python .input}
train_data.head()
```

Let's visualize a matching image pair.

```{.python .input}
pil_img = Image(filename=train_data[image_col_1][5])
display(pil_img)
```
```{.python .input}
pil_img = Image(filename=train_data[image_col_2][5])
display(pil_img)
```

Here are two images that do not match.
```{.python .input}
pil_img = Image(filename=train_data[image_col_1][0])
display(pil_img)
```
```{.python .input}
pil_img = Image(filename=train_data[image_col_2][0])
display(pil_img)
```


## Train your Model

Ideally, we want to obtain a model that can return high/low scores for positive/negative image pairs. With AutoMM, we can easily train a model that captures the semantic relationship between images. Bascially, it uses [Swin Transformer](https://arxiv.org/abs/2103.14030) to project each image into a high-dimensional vector and compute the cosine similarity of feature vectors. 

With AutoMM, you just need to specify the `query`, `response`, and `label` column names and fit the model on the training dataset without worrying the implementation details.

```{.python .input}
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(
        problem_type="image_similarity",
        query=image_col_1, # the column name of the first image
        response=image_col_2, # the column name of the second image
        label=label_col, # the label column name
        match_label=match_label, # the label indicating that query and response have the same semantic meanings.
        eval_metric='auc', # the evaluation metric
    )
    
# Fit the model
predictor.fit(
    train_data=train_data,
    time_limit=180,
)
```

## Evaluate on Test Dataset
You can evaluate the predictor on the test dataset to see how it performs with the roc_auc score:

```{.python .input}
score = predictor.evaluate(test_data)
print("evaluation score: ", score)
```

## Predict on Image Pairs
Given new image pairs, we can predict whether they match or not.
```{.python .input}
pred = predictor.predict(test_data.head(3))
print(pred)
```
The predictions use a naive probability threshold 0.5. That is, we choose the label with the probability larger than 0.5.

## Predict Matching Probabilities
However, you can do more customized thresholding by getting probabilities.
```{.python .input}
proba = predictor.predict_proba(test_data.head(3))
print(proba)
```

## Extract Embeddings
You can also extract embeddings for each image of a pair.
```{.python .input}
embeddings_1 = predictor.extract_embedding({image_col_1: test_data[image_col_1][:5].tolist()})
print(embeddings_1.shape)
embeddings_2 = predictor.extract_embedding({image_col_2: test_data[image_col_2][:5].tolist()})
print(embeddings_2.shape)
```


## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.


## Customization

To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
