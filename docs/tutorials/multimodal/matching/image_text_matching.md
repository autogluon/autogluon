# Image<-->Text Semantic Matching with AutoMM
:label:`image_text_matching`

Vision and language are two important aspects of human intelligence to understand the real world. Image-text matching, measuring the visual-semantic
similarity between image and text, plays a critical role in bridging the vision and language. 
Learning a joint space where text
and image feature vectors are aligned is a typical solution for image-text matching. It is becoming increasingly significant for various vision-and-language tasks,
such as cross-modal retrieval, image
captioning, text-to-image synthesis, and multimodal neural machine translation. This tutorial will introduce how to apply AutoMM to the image-text matching task.


```{.python .input}
import os
import warnings
from IPython.display import Image, display
import numpy as np
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset

In this tutorial, we will use the Flickr30K dataset to demonstrate the image-text matching.
The Flickr30k dataset is a popular benchmark for sentence-based picture portrayal. The dataset is comprised of 31,783 images that capture people engaged in everyday activities and events. Each image has a descriptive caption. We organized the dataset using pandas dataframe. To get started, Let's download the dataset.


```{.python .input}
from autogluon.core.utils.loaders import load_pd
import pandas as pd
download_dir = './ag_automm_tutorial_imgtxt'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/flickr30k.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

Then we will load the csv files.


```{.python .input}
dataset_path = os.path.join(download_dir, 'flickr30k_processed')
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
val_data = pd.read_csv(f'{dataset_path}/val.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col = "image"
text_col = "caption"
```

We also need to expand the relative image paths to use their absolute local paths.


```{.python .input}
def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
val_data[image_col] = val_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

Take `train_data` for example, let's see how the data look like in the dataframe.


```{.python .input}
train_data.head()
```

Each row is one image and text pair, implying that they match each other. Since one image corresponds to five captions in the dataset, we copy each image path five times to build the correspondences. We can visualize one image-text pair.


```{.python .input}
train_data[text_col][0]
```


```{.python .input}
pil_img = Image(filename=train_data[image_col][0])
display(pil_img)
```

To perform evaluation or semantic search, we need to extract the unique image and text items from `text_data` and add one label column in the `test_data`.


```{.python .input}
test_image_data = pd.DataFrame({image_col: test_data[image_col].unique().tolist()})
test_text_data = pd.DataFrame({text_col: test_data[text_col].unique().tolist()})
test_data_with_label = test_data.copy()
test_label_col = "relevance"
test_data_with_label[test_label_col] = [1] * len(test_data)
```

## Initialize Predictor
To initialize a predictor for image-text matching, we need to set `problem_type` as `image_text_similarity`. `query` and `response` refer to the two dataframe columns in which two items in one row should match each other. You can set `query=text_col` and `response=image_col`, or `query=image_col` and `response=text_col`. In image-text matching, `query` and `response` are equivalent.


```{.python .input}
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(
            query=text_col,
            response=image_col,
            problem_type="image_text_similarity",
            eval_metric="recall",
        )
```

By initializing the predictor for `image_text_similarity`, you have loaded the pretrained CLIP backbone `openai/clip-vit-base-patch32`.

## Directly Evaluate on Test Dataset (Zero-shot)
You may be interested in getting the pretrained model's performance on your data. Let's compute the text-to-image and image-to-text retrieval scores.


```{.python .input}
txt_to_img_scores = predictor.evaluate(
            data=test_data_with_label,
            query_data=test_text_data,
            response_data=test_image_data,
            label=test_label_col,
            cutoffs=[1, 5, 10],
        )
img_to_txt_scores = predictor.evaluate(
            data=test_data_with_label,
            query_data=test_image_data,
            response_data=test_text_data,
            label=test_label_col,
            cutoffs=[1, 5, 10],
        )
print(f"txt_to_img_scores: {txt_to_img_scores}")
print(f"img_to_txt_scores: {img_to_txt_scores}")
```

Here we report the `recall`, which is the `eval_metric` in intializing the predictor above. One `cutoff` value means using the top k retrieved items to calculate the score. You may find that the text-to-image recalls are much higher than the image-to-text recalls. This is because each image is paired with five texts. In image-to-text retrieval, the upper bound of `recall@1` is 20%, which means that the top-1 text is correct, but there are totally five texts to retrieve.

## Finetune Predictor
After measuring the pretrained performance, we can finetune the model on our dataset to see whether we can get improvements. For a quick demo, here we set the time limit to 180 seconds.


```{.python .input}
predictor.fit(
            train_data=train_data,
            tuning_data=val_data,
            time_limit=180,
        )
```

## Evaluate the Finetuned Model on the Test Dataset
Now Let's evaluate the finetuned model. Similarly, we also compute the recalls of text-to-image and image-to-text retrievals.


```{.python .input}
txt_to_img_scores = predictor.evaluate(
            data=test_data_with_label,
            query_data=test_text_data,
            response_data=test_image_data,
            label=test_label_col,
            cutoffs=[1, 5, 10],
        )
img_to_txt_scores = predictor.evaluate(
            data=test_data_with_label,
            query_data=test_image_data,
            response_data=test_text_data,
            label=test_label_col,
            cutoffs=[1, 5, 10],
        )
print(f"txt_to_img_scores: {txt_to_img_scores}")
print(f"img_to_txt_scores: {img_to_txt_scores}")
```

We can observe large improvements over the zero-shot predictor. This means that finetuning CLIP on our customized data may help achieve better performance.

## Predict Whether Image and Text Match
Whether finetuned or not, the predictor can predict whether image and text pairs match.


```{.python .input}
pred = predictor.predict(test_data.head(5))
print(pred)
```

## Predict Matching Probabilities
The predictor can also return to you the matching probabilities.


```{.python .input}
proba = predictor.predict_proba(test_data.head(5))
print(proba)
```

The second column is the probability of being a match.

## Extract Embeddings
Another common user case is to extract image and text embeddings.


```{.python .input}
image_embeddings = predictor.extract_embedding({image_col: test_image_data[image_col][:5].tolist()})
print(image_embeddings.shape) 
```


```{.python .input}
text_embeddings = predictor.extract_embedding({text_col: test_text_data[text_col][:5].tolist()})
print(text_embeddings.shape)
```

## Semantic Search
We also provide an advanced util function to conduct semantic search. First, given one or more texts, we can search semantically similar images from an image database.


```{.python .input}
from autogluon.multimodal.utils import semantic_search
text_to_image_hits = semantic_search(
        matcher=predictor,
        query_data=test_text_data.iloc[[3]],
        response_data=test_image_data,
        top_k=5,
    )
```

Let's visualize the text query and top-1 image response.


```{.python .input}
test_text_data.iloc[[3]]
```


```{.python .input}
pil_img = Image(filename=test_image_data[image_col][text_to_image_hits[0][0]['response_id']])
display(pil_img)
```

Similarly, given one or more images, we can retrieve texts with similar semantic meanings.


```{.python .input}
image_to_text_hits = semantic_search(
        matcher=predictor,
        query_data=test_image_data.iloc[[6]],
        response_data=test_text_data,
        top_k=5,
    )
```

Let's visualize the image query and top-1 text response.


```{.python .input}
pil_img = Image(filename=test_image_data[image_col][6])
display(pil_img)
```


```{.python .input}
test_text_data[text_col][image_to_text_hits[0][1]['response_id']]
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.


## Customization

To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
