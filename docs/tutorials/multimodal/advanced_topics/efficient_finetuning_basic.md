# Single GPU Billion-scale Model Training via Parameter-Efficient Finetuning
:label:`sec_automm_efficient_finetuning_basic`

As pointed out by [a recent paper from Stanford Institute for Human-Centered Artificial Intelligence](https://arxiv.org/pdf/2108.07258.pdf), 
AI is undergoing a paradigm shift with the rise of "foundation models", i.e., giant models that are trained on a diverse collection of datasets generally in a self-supervised way. 
These foundation models, which are the key of AutoMM, can be easily adapted to down-stream applications. However, as the size of these foundation models grows, finetuning these models becomes increasingly difficult. 
Following is a figure from the [Microsoft research blog](https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) that demonstrates the trend:

![Scaling of foundation models](https://www.microsoft.com/en-us/research/uploads/prod/2021/10/model-size-graph.jpg)
:width:`500px`

The goal of AutoMM is to help anyone solve machine learning problems via open source foundation models, including these giant models. 
To finetune these large-scale models, we adopt the recently popularized **parameter-efficient finetuning** technique. 
The idea is to either finetune a small subset of the weights in the foundation model (e.g., [BitFit](https://aclanthology.org/2022.acl-short.1.pdf)), 
or adding a tiny tunable structure on top of the fixed backbone (e.g., [Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243.pdf),
[LoRA](https://arxiv.org/pdf/2106.09685.pdf), [Adapter](https://arxiv.org/abs/1902.00751), [MAM Adapter](https://arxiv.org/pdf/2110.04366.pdf), [IA^3](https://arxiv.org/abs/2205.05638)). 
These techniques can effectively reduce the peak memory usage and model training time, while maintaining the performance.

In this tutorial, we introduce how to apply parameter-efficient finetuning in MultiModalPredictor. 
We first introduce how to adopt the `"ia3_bias"` algorithm for parameter-efficient finetuning. Afterwards, we show how you can simply combine `"ia3_bias"` 
and gradient checkpointing to finetune the XL-variant of Google's [FLAN-T5](https://arxiv.org/abs/2210.11416) via a single NVIDIA T4 GPU. 


## Prepare Dataset

The [Cross-Lingual Amazon Product Review Sentiment](https://webis.de/data/webis-cls-10.html) dataset contains Amazon product reviews in four languages. 
Here, we load the English and German fold of the dataset. In the label column, `0` means negative sentiment and `1` means positive sentiment. 
For the purpose of demonstration, we downsampled the training data to 1000 samples. We will train the model on the English dataset and 
directly evaluate its performance on the German and Japanese test set.


```{.python .input}
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/multilingual-datasets/amazon_review_sentiment_cross_lingual.zip -O amazon_review_sentiment_cross_lingual.zip
!unzip -q -o amazon_review_sentiment_cross_lingual.zip -d .
```


```{.python .input}
import os
import shutil
os.environ["TRANSFORMERS_CACHE"] = "cache"

def clear_cache():
    if os.path.exists("cache"):
        shutil.rmtree("cache")

clear_cache()
```


```{.python .input}
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

train_en_df = pd.read_csv("amazon_review_sentiment_cross_lingual/en_train.tsv",
                          sep="\t",
                          header=None,
                          names=["label", "text"]) \
                .sample(1000, random_state=123).reset_index(drop=True)

test_en_df = pd.read_csv("amazon_review_sentiment_cross_lingual/en_test.tsv",
                          sep="\t",
                          header=None,
                          names=["label", "text"]) \
               .sample(200, random_state=123).reset_index(drop=True)
test_de_df = pd.read_csv("amazon_review_sentiment_cross_lingual/de_test.tsv",
                          sep="\t", header=None, names=["label", "text"]) \
               .sample(200, random_state=123).reset_index(drop=True)

test_jp_df = pd.read_csv('amazon_review_sentiment_cross_lingual/jp_test.tsv',
                          sep='\t', header=None, names=['label', 'text']) \
               .sample(200, random_state=123).reset_index(drop=True)
train_en_df.head(5)
```

```{.python .input}
test_jp_df.head(5)
```

## Finetuning Multilingual Model with IA3 + BitFit

In AutoMM, to enable efficient finetuning, just specify the `optimization.efficient_finetune` to be `"ia3_bias"`.


```{.python .input}
from autogluon.multimodal import MultiModalPredictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-multilingual_ia3"
predictor = MultiModalPredictor(label="label",
                                path=model_path)
predictor.fit(train_en_df,
              presets="multilingual",
              hyperparameters={
                  "optimization.efficient_finetune": "ia3_bias",
                  "optimization.lr_decay": 0.9,
                  "optimization.learning_rate": 3e-03,
                  "optimization.end_lr": 3e-03,
                  "optimization.max_epochs": 3,
                  "optimization.warmup_steps": 0,
                  "env.batch_size": 32,
              })
```

The fraction of the tunable parameters is around **0.5%** of all parameters. Actually, the model trained purely on English data can achieve good performance 
on the test sets, even on the German / Japanese test set. It obtained **comparable results** as full-finetuning as in :ref:`sec_automm_textprediction_multilingual`.


```{.python .input}
score_in_en = predictor.evaluate(test_en_df)
score_in_de = predictor.evaluate(test_de_df)
score_in_jp = predictor.evaluate(test_jp_df)
print('Score in the English Testset:', score_in_en)
print('Score in the German Testset:', score_in_de)
print('Score in the Japanese Testset:', score_in_jp)
```

## Training FLAN-T5-XL on Single GPU

By combining [gradient checkpointing](https://pytorch.org/docs/stable/checkpoint.html) and parameter-efficient finetuning, it is feasible to finetune 
[google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl) that has close to two billion parameterswith a single T4 GPU available in
[AWS G4 instances](https://aws.amazon.com/ec2/instance-types/g4/). 
To turn on gradient checkpointing, you just need to set `"model.hf_text.gradient_checkpointing"` to `True`. 
To accelerate the training, we downsample the number of training sampels to be 200.


```{.python .input}
# Just for clean the space
clear_cache()
shutil.rmtree(model_path)
```


```python
from autogluon.multimodal import MultiModalPredictor

train_en_df_downsample = train_en_df.sample(200, random_state=123)

new_model_path = f"./tmp/{uuid.uuid4().hex}-multilingual_ia3_gradient_checkpoint"
predictor = MultiModalPredictor(label="label",
                                path=new_model_path)
predictor.fit(train_en_df_downsample,
              presets="multilingual",
              hyperparameters={
                  "model.hf_text.checkpoint_name": "google/flan-t5-xl",
                  "model.hf_text.gradient_checkpointing": True,
                  "model.hf_text.low_cpu_mem_usage": True,
                  "optimization.efficient_finetune": "ia3_bias",
                  "optimization.lr_decay": 0.9,
                  "optimization.learning_rate": 3e-03,
                  "optimization.end_lr": 3e-03,
                  "optimization.max_epochs": 1,
                  "optimization.warmup_steps": 0,
                  "env.batch_size": 1,
                  "env.eval_batch_size_ratio": 1
              })

```

```
Global seed set to 123
Auto select gpus: [0]
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name              | Type                         | Params
-------------------------------------------------------------------
0 | model             | HFAutoModelForTextPrediction | 1.2 B 
1 | validation_metric | AUROC                        | 0     
2 | loss_func         | CrossEntropyLoss             | 0     
-------------------------------------------------------------------
203 K     Trainable params
1.2 B     Non-trainable params
1.2 B     Total params
4,894.913 Total estimated model params size (MB)
Epoch 0, global step 20: 'val_roc_auc' reached 0.88802 (best 0.88802), saving model to '/home/ubuntu/autogluon/docs/tutorials/multimodal/advanced_topics/multilingual_ia3_gradient_checkpoint/epoch=0-step=20.ckpt' as top 1
Epoch 0, global step 40: 'val_roc_auc' reached 0.94531 (best 0.94531), saving model to '/home/ubuntu/autogluon/docs/tutorials/multimodal/advanced_topics/multilingual_ia3_gradient_checkpoint/epoch=0-step=40.ckpt' as top 1
`Trainer.fit` stopped: `max_epochs=1` reached.





<autogluon.multimodal.predictor.MultiModalPredictor at 0x7fd58c4dbca0>
```



```python
score_in_en = predictor.evaluate(test_en_df)
print('Score in the English Testset:', score_in_en)
```

```
Score in the English Testset: {'roc_auc': 0.931263189629183}
```


```python
# Just for clean the space
clear_cache()
shutil.rmtree(new_model_path)
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
