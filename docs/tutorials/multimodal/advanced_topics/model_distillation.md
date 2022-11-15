# Knowledge Distillation in AutoMM
:label:`sec_automm_distillation_multilingual`

Pretrained foundation models are becoming increasingly large. However, these models are difficult to deploy due to 
limited resources available in deployment scenarios. To benefit from large models under this constraint, 
you transfer the knowledge from the large-scale teacher models to the student model, with knowledge distillation.
In this way, the small student model can be practically deployed under real-world scenarios,
while the performance will be better than training the student model from scratch thanks to the teacher.

In this tutorial, we introduce how to adopt `MultiModalPredictor` for knowledge distillation. For the purpose of demonstration, we use the [Question-answering NLI](https://paperswithcode.com/dataset/qnli) dataset, 
which comprises 104,743 question, answer pairs sampled from question answering datasets. We will demonstrate how to use a large model to guide the learning and improve the performance of a small model in AutoGluon.

## Load Dataset

The [Question-answering NLI](https://paperswithcode.com/dataset/qnli) dataset contains 
sentence pairs in English. In the label column, `0` means that the sentence is not related to the question and `1` means that the sentence is related to the question.


```{.python .input}
from datasets import load_dataset

dataset = load_dataset("glue", "qnli")
```


```{.python .input}
dataset['train']
```


```{.python .input}
from sklearn.model_selection import train_test_split

train_valid_df = dataset["train"].to_pandas()[["question", "sentence", "label"]].sample(1000, random_state=123)
train_df, valid_df = train_test_split(train_valid_df, test_size=0.2, random_state=123)
test_df = dataset["validation"].to_pandas()[["question", "sentence", "label"]].sample(1000, random_state=123)
```

## Load the Teacher Model

In our example, we will directly load a teacher model with the [google/bert_uncased_L-12_H-768_A-12](https://huggingface.co/google/bert_uncased_L-12_H-768_A-12) backbone that has been trained on QNLI and distill it into a student model with the [google/bert_uncased_L-6_H-768_A-12](https://huggingface.co/google/bert_uncased_L-6_H-768_A-12) backbone.


```{.python .input}
!wget --quiet https://automl-mm-bench.s3.amazonaws.com/unit-tests/distillation_sample_teacher.zip -O distillation_sample_teacher.zip
!unzip -q -o distillation_sample_teacher.zip -d .
```


```{.python .input}
from autogluon.multimodal import MultiModalPredictor

teacher_predictor = MultiModalPredictor.load("ag_distillation_sample_teacher/")
```

## Distill to Student

Training the student model is straight forward. You may just add the `teacher_predictor` argument when calling `.fit()`.


```{.python .input}
student_predictor = MultiModalPredictor(label="label")
student_predictor.fit(
    train_df,
    tuning_data=valid_df,
    teacher_predictor=teacher_predictor,
    hyperparameters={
        "model.hf_text.checkpoint_name": "google/bert_uncased_L-6_H-768_A-12",
        "optimization.max_epochs": 2,
    }
)
```


```{.python .input}
print(student_predictor.evaluate(data=test_df))
```

## Comparing with Direct Finetuning

We then finetune a small model [mMiniLMv2](https://arxiv.org/abs/2012.15828) without distillation. 
We can still load the multilingual MiniLMv2 model from Huggingface/Transformers, 
with the key as [nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large](ahttps://huggingface.co/nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large). 
To simplify the experiment, we also just finetune for 4 epochs.


```{.python .input}
nodistill_predictor = MultiModalPredictor(label="label")
nodistill_predictor.fit(
    train_df,
    tuning_data=valid_df,
    hyperparameters={
        "model.hf_text.checkpoint_name": "google/bert_uncased_L-6_H-768_A-12",
        "optimization.max_epochs": 2,
    }
)
```


```{.python .input}
print(nodistill_predictor.evaluate(data=test_df))
```

We can find that via knowledge distillation, the performance of `student_predictor` is better than `nodistill_predictor`.

## More about Knowledge Distillation

To learn how to customize distillation, see the distillation examples 
and README in [AutoMM Distillation Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm/distillation).
Especially the [multilingual distillation example](https://github.com/awslabs/autogluon/tree/master/examples/automm/distillation/automm_distillation_pawsx.py) with more details and customization.

## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
