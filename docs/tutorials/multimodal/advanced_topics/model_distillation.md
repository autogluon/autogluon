# Knowledge Distillation in AutoMM
:label:`sec_automm_distillation_multilingual`

Pretrained foundation models are becoming increasingly large. However, these models are difficult to deploy due to 
limited resources available in deployment scenarios. To benefit from large models under this constraint, 
you transfer the knowledge from the large-scale teacher models to the student model, with knowledge distillation.
In this way, the small student model can be practically deployed under real-world scenarios,
while the performance will be better than training the student model from scratch thanks to the teacher.

In this tutorial, we introduce how to adopt `MultiModalPredictor` for knowledge distillation. For the purpose of demonstration, 
we use the [Question-answering NLI](https://paperswithcode.com/dataset/qnli) dataset, 
which comprises 23,659 human translated PAWS evaluation pairs in six typologically distinct languages:
French, Spanish, German, Chinese, Japanese, and Korean. 
We will demonstrate how to use a large model to guide the learning 
and improve the performance of a small model in AutoGluon.

## Load Dataset

The [Cross-Lingual Paraphrase Adversaries from Word Scrambling (PAWS-X)](https://arxiv.org/pdf/1908.11828.pdf) dataset contains 
sentence pairs in six languages. 
Here, we use the English and German to demonstrate.
In the label column, `0` means different meaning for the pair of sentences and `1` means same meaning.

```
from autogluon.multimodal import MultiModalPredictor
from datasets import load_dataset
from time import time
import pandas as pd

def getDatasetSplits(pawsx_tasks = ["en", "de"]):
    train_dfs = {}
    val_dfs = {}
    test_dfs = {}
    for task in pawsx_tasks:
        dataset = load_dataset("paws-x", task)
        train_dfs[task] = dataset["train"].to_pandas()
        val_dfs[task] = dataset["validation"].to_pandas()
        test_dfs[task] = dataset["test"].to_pandas()
        print(
            "task %s: train %d, val %d, test %d"
            % (task, len(train_dfs[task]), len(val_dfs[task]), len(test_dfs[task]))
        )
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_dfs["all"] = pd.concat(test_dfs)

    return train_df, val_df, test_dfs
    
train_df, val_df, test_dfs = getDatasetSplits()
```

## Finetune the large teacher model

We first finetune a large teacher model [DeBERTaV3](https://arxiv.org/abs/2111.09543). 
Since `MultiModalPredictor` integrates with the [Huggingface/Transformers](https://huggingface.co/docs/transformers/index) 
(as explained in :ref:`sec_textprediction_customization`), 
we directly load the multilingual DeBERTaV3 model pretrained by Microsoft from Huggingface/Transformers, 
with the key as [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base).

```
teacher_predictor = MultiModalPredictor(label="label", eval_metric="accuracy")
teacher_predictor.fit(
    train_df,
    tuning_data=val_df,
    hyperparameters={
        "model.hf_text.checkpoint_name": "microsoft/mdeberta-v3-base",
        "env.precision": "32",
        "optimization.max_epochs": 8,
    }
)
```

```
teacher_result = {}
start = time()
for test_name, test_df in test_dfs.items():
    teacher_result[test_name] = teacher_predictor.evaluate(data=test_df, metrics="accuracy")
teacher_usedtime = time() - start
print('Time used by the teacher model for inference:')
print(teacher_usedtime)
```

```
print('Result of the teacher model on the German Testset:')
print(teacher_result["de"])
```

```
print('Result of the teacher model on the English Testset:')
print(teacher_result["en"])
```

## Finetune a small model without distillation

We then finetune a small model [mMiniLMv2](https://arxiv.org/abs/2012.15828) without distillation. 
We can still load the multilingual MiniLMv2 model from Huggingface/Transformers, 
with the key as [nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large](ahttps://huggingface.co/nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large). 
To simplify the experiment, we also just finetune for 4 epochs.

```
nodistill_predictor = MultiModalPredictor(label="label", eval_metric="accuracy")
nodistill_predictor.fit(
    train_df,
    tuning_data=val_df,
    hyperparameters={
        "model.hf_text.checkpoint_name": "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large",
        "env.precision": "32",
        "optimization.max_epochs": 8,
    }
)
```

```
nodistill_result = {}
start = time()
for test_name, test_df in test_dfs.items():
    nodistill_result[test_name] = nodistill_predictor.evaluate(data=test_df, metrics="accuracy")
nodistill_usedtime = time() - start
print('Time used without distillation for inference:')
print(nodistill_usedtime)
```

```
print('Result without distillation on the German Testset:')
print(nodistill_result["de"])
```

```
print('Result without distillation on the English Testset:')
print(nodistill_result["en"])
```


We can find that the model can achieve faster speed on the test dataset
but also results in a degradation in performance. 
Next, we will show how to enable distillation so we can get better performance with this small model.

## Finetune a small model with distillation

To improve the small model's performance without increasing the
computational cost during inference, we can apply knowledge distillation, 
i.e. using the large model to guide the training of the small model.

```
student_predictor = MultiModalPredictor(label="label", eval_metric="accuracy")
student_predictor.fit(
    train_df,
    tuning_data=val_df,
    hyperparameters={
        "model.hf_text.checkpoint_name": "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large",
        "env.precision": "32",
        "optimization.max_epochs": 8,
    },
    teacher_predictor=teacher_predictor,
)
```

```
student_result = {}
start = time()
for test_name, test_df in test_dfs.items():
    student_result[test_name] = student_predictor.evaluate(data=test_df, metrics="accuracy")
student_usedtime = time() - start
print('Time used with distillation for inference:')
print(student_usedtime)
```

```
print('Result with distillation on the German Testset:')
print(student_result["de"])
```

```
print('Result with distillation on the English Testset:')
print(student_result["en"])
```

As we can see, the student model is still fast at inference 
but with higher accuracy in both English and German.

## Customize Knowledge Distillation

To learn how to customize distillation, see the distillation examples 
and README in [AutoMM Distillation Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm/distillation).
Especially the [multilingual distillation example](https://github.com/awslabs/autogluon/tree/master/examples/automm/distillation/automm_distillation_pawsx.py) with more details and customization.

## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.
