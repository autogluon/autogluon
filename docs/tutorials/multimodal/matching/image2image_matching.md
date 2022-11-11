# Image-to-Image Matching with AutoMM 
:label:`image2image_matching`

Computing the similarity between two images is a common task in computer vision, with several practical applications such as detecting same or different product, etc. In general, image similarity models will take two images as input and transform them into vectors, and then similarity scores calculated using cosine similarity, dot product, or Euclidean distances are used to measure how alike or different of the two images. 

## Prepare your Data
In this tutorial, we will demonstrate how to use AutoMM for image-to-image matching with the simplified Stanford Online Products dataset ([SOP](https://cvgl.stanford.edu/projects/lifted_struct/)). The data can be downloaded from [here](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip). Directly unzipping the data will be enough. 

Stanford Online Products dataset is introduced for metric learning. There are 12 categories of products in this dataset: bicycle, cabinet, chair, coffee maker, fan, kettle, lamp, mug, sofa, stapler, table and toaster. Each category has some products, and each product has several images captured from different views. Here, we consider different views of the same product as positive pairs (labeled as 1) and images from different products as negative pairs (labeled as 0). 

The following code downloads and loads the annotation into dataframes. Please make sure these annotation files are saved to the same directory where you store the actual data. 

```{.python .input}
from autogluon.core.utils.loaders import load_pd
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sop_train = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/stanford_online_products/sop_train.csv')
sop_test = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/stanford_online_products/sop_test.csv')
print(sop_train.head())
```

## Train your Model

Ideally, we want to obtain a model that can return high/low scores for positive/negative image pairs. With AutoMM, we can easily train a model that captures the semantic relationship between images. Bascially, it uses [Swin Transformer](https://arxiv.org/abs/2103.14030) to project each image into a high-dimensional vector and treat the matching problem as a classification problem. 

With AutoMM, you just need to specify the query, response, and label column names and fit the model on the training dataset without worrying the implementation details.

```python
from autogluon.multimodal import MultiModalPredictor

# Initialize the model
matcher = MultiModalPredictor(
        problem_type="image_similarity",
        query="Image1", # the column name of the first image
        response="Image2", # the column name of the second image
        label="Label", # the label column name
        eval_metric='auc', # the evaluation metric
    )

# Fit the model
matcher.fit(
    train_data=sop_train,
    time_limit=180,
)
```

```
Global seed set to 123
Auto select gpus: [0, 1, 2, 3]
Using 16bit native Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/4
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/4
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/4

  | Name              | Type                            | Params
----------------------------------------------------------------------
0 | query_model       | TimmAutoModelForImagePrediction | 86.7 M
1 | response_model    | TimmAutoModelForImagePrediction | 86.7 M
2 | validation_metric | AUROC                           | 0     
3 | loss_func         | ContrastiveLoss                 | 0     
4 | miner_func        | PairMarginMiner                 | 0     
----------------------------------------------------------------------
86.7 M    Trainable params
0         Non-trainable params
86.7 M    Total params
173.486   Total estimated model params size (MB)

Epoch 0:   0%|▍                                                                                                                               | 3/779 [00:23<1:39:41,  7.71s/it, loss=0.662, v_num=]
Epoch 0:  50%|███████████████████████████████████████████████████████████████▉                                                                | 389/779 [02:23<02:23,  2.72it/s, loss=0.671, v_num=Epoch 0, global step 79: 'val_roc_auc' reached 0.87266 (best 0.87266), saving model to '/home/ubuntu/data/img2img_matching/Stanford_Online_Products/AutogluonModels/ag-20221110_195408/epoch=0-step=79.ckpt' as top 3
Epoch 0:  73%|██████████████████████████████████████████████████████████████████████████████████████████████▋                                  | 572/779 [03:38<01:19,  2.62it/s, loss=0.63, v_num=]

```

## Evaluate on Test Dataset
You can evaluate the macther on the test dataset to see how it performs with the roc_auc score:

```python
score = matcher.evaluate(sop_test)
print("evaluation score: ", score)
```

```
Predicting DataLoader 0: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:12<00:00,  1.46it/s]
Predicting DataLoader 0: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [00:11<00:00,  1.57it/s]
Predicting DataLoader 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 177/177 [01:48<00:00,  1.63it/s]
evaluation score:  {'roc_auc': 0.8907004748329923}
```


## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.


## Customization

To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
