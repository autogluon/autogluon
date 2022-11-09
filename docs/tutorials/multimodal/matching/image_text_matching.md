# Image<-->Text Matching with AutoMM - Quick Start
:label:`image_text_matching`

Vision and language are two important aspects of human intelligence to understand the real world. Image-text matching plays a critical role in bridging the vision and language. 
Image-text matching refers to measuring the visual-semantic
similarity between image and text, for which learning a joint space where text
and image feature vectors are aligned is a typical solution. Image-text matching is becoming increasingly significant for various vision-and-language tasks,
such as cross-modal retrieval, image
captioning, text-to-image synthesis, and multimodal neural machine translation. This tutorial will introduce how to apply MultiModalPredictor to the image-text matching task.


```{.python .input}
import os
import numpy as np
import warnings
from IPython.display import Image, display
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset

In this tutorial, we will use the Flickr30K dataset to demonstrate the image-text matching.
The Flickr30k dataset is a popular benchmark for sentence-based picture portrayal. The dataset is comprised of 31,783 images that capture people engaged in everyday activities and events. Each image has a descriptive caption. We re-organize the dataset using pandas dataframe. To get started, Let's download the dataset. 


```python
from autogluon.core.utils.loaders import load_pd
import pandas as pd
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/flickr30k.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

Then we will load the csv files.


```python
dataset_path = os.path.join(download_dir, '/flickr30k_processed')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
val_data = pd.read_csv(f'{dataset_path}/val.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col = "image"
text_col = "caption"
```

We also need to expand the relative image paths to use their absolute local paths.


```python
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
val_data[image_col] = val_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

train_data[image_col].iloc[0]
```

Take train_data for example, let's see how the data look like in the table.


```python
train_data.head()
```

Each row is one image and text pair, implying that they match each other. Since one image corresponds to five captions in the dataset, we copy each image path five times to build the correspondences.

We can visualize one image-text pair.


```python
train_data[text_col][0]
```


```python
pil_img = Image(filename=train_data[image_col][0])
display(pil_img)
```

To evaluate on test data easily, we need to convert it a little bit.


```python
test_image_data = pd.DataFrame({image_col: test_data[image_col].unique().tolist()})
test_text_data = pd.DataFrame({text_col: test_data[text_col].unique().tolist()})
test_data_with_label = test_data.copy()
test_label_col = "relevance"
test_data_with_label[test_label_col] = [1] * len(test_data)
```

## Initialize Predictor


```python
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(
            query=text_col,
            response=image_col,
            pipeline="image_text_similarity",
            eval_metric="recall",
        )
```

## Directly Evaluate on Test Dataset (Zero-shot)


```python
txt_to_img_scores = predictor.evaluate(
            data=test_data_with_label,
            query_data=test_text_data,
            response_data=test_image_data,
            label=test_label_column,
            cutoffs=[1, 5, 10],
        )
img_to_txt_scores = predictor.evaluate(
            data=test_data_with_label,
            query_data=test_image_data,
            response_data=test_text_data,
            label=test_label_column,
            cutoffs=[1, 5, 10],
        )
print(f"txt_to_img_scores: {txt_to_img_scores}")
print(f"img_to_txt_scores: {img_to_txt_scores}")
```

## Finetune Predictor


```python
predictor.fit(
            train_data=train_data,
            tuning_data=val_data,
            time_limit=180,
        )
```

## Evaluate on the Test Dataset (Finetuned)


```python
txt_to_img_scores = predictor.evaluate(
            data=test_df_with_label,
            query_data=test_query_text_data,
            response_data=test_response_image_data,
            label=test_label_column,
            cutoffs=[1, 5, 10],
        )
img_to_txt_scores = predictor.evaluate(
            data=test_df_with_label,
            query_data=test_query_image_data,
            response_data=test_response_text_data,
            label=test_label_column,
            cutoffs=[1, 5, 10],
        )
print(f"txt_to_img_scores: {txt_to_img_scores}")
print(f"img_to_txt_scores: {img_to_txt_scores}")
```

We can observe obvious improvements over the zero-shot predictor.

## Predict Whether Image and Text Match


```python
pred = predictor.predict(test_data.head(5))
```

## Predict Matching Probabilities


```python
pred = predictor.predict_proba(test_data.head(5))
```

## Extract Embeddings 


```python
image_embeddings = predictor.extract_embedding({image_col: test_image_data[image_col][:5].tolist()})
print(image_embeddings.shape) 
```


```python
text_embeddings = predictor.extract_embedding({text_col: test_text_data[text_col][:5].tolist()})
print(text_embeddings.shape)
```

## Semantic Search

Text-to-image retrieval.


```python
text_to_image_hits = semantic_search(
        matcher=predictor,
        query_data=test_text_data.head(1),
        response_data=test_image_data,
        top_k=5,
    )
```

Image-to-text retrieval.


```python
image_to_text_hits = semantic_search(
        matcher=predictor,
        query_data=test_image_data.head(1),
        response_data=test_text_data,
        top_k=5,
    )
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.


## Customization

To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
