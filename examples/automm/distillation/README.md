# Model Distillation in AutoMM

Examples showing how to use `MultiModalPredictor` for model distillation.

## 1. Distillation on GLUE and PAWS-X tasks

### 1.1 Example

[`automm_distillation_glue.py`](./automm_distillation_glue.py) : This example provides a use case for GLUE tasks

[`automm_distillation_pawsx.py`](./automm_distillation_pawsx.py) : This example provides a use case for PAWS_X

To run the example:

`python automm_distillation_<task_name>.py --<flag> <value>`

- `glue_task` (only for GLUE example) determines to run the experiments on which GLUE task, refers to [Dataset Section](###1.2-Datasets).
- `pawsx_teacher_tasks` (only for PAWS-X example) determines which language data to use while training the teacher model
- `pawsx_student_tasks` (only for PAWS-X example) determines which language data to use while training the student model
- `teacher_model` is the name of teacher model
- `student_model` is the name of the student model. It is recommended to have the same backbone as teacher model (e.g. both ELECTRA or both BERT).
- `seed` determines the random seed. Default is 123.
- `precision` determines precision of the GPU operations
- `max_epochs` determines the max epoch for training student model.
- `time_limit` determines the max time to train each model. The unit is second.
- `num_gpu` determines num of gpu used to train the models. Default is -1 which means using all.
- `temperature` determines the temperature for soft cross entropy.
- `hard_label_weight` determines the weight of hard label loss (ground truth logits).
- `soft_label_weight` determines the weight of soft label loss (teacher model's output).
- `softmax_regression_weight` determines the weight of softmax regression loss
- `output_feature_loss_weight` determines the weight of output_feature loss
- `rkd_distance_loss_weight` determines the weight of RKD distance loss
- `rkd_angle_loss_weight` determines the weight of RKD angle loss
- `soft_label_loss_type` determines the loss function of soft label loss
- `softmax_regression_loss_type` determines the loss function of softmax regression loss
- `output_feature_loss_type` determines the loss function of output_feature loss, currently support "mean_square_error" and "cosine_distance"
- `finetuned_model_cache_folder` is the path to cache models trained without distillation.
- `resume` if True, models without distillation will be loaded from cache.

### 1.2 Dataset

#### GLUE

We borrow 7 NLP tasks in GLUE [1], and use identically the same abbreviation as in [1] to name each tasks,
i.e. "mnli(m/mm)", "qqp", "qnli", "sst2", "stsb", "mrpc", and "rte".
The dataset are loaded using huggingface's API: https://huggingface.co/datasets/glue.
All Data will be automatically downloaded from HuggingFace (thus online connection is necessary) if it does not exist with the given dataset path.

In GLUE, labels in test sets are private and thus we use training set for training/validation and use validation set for testing.

#### PAWS-X

Cross-lingual PAWS (PAWS-X) [3] is an extension of the Wikipedia portion
of the PAWS evaluation and test examples to six languages:
Spanish, French, German, Chinese, Japanese, and Korean.
This new corpus consists of 23,659 human translated example pairs
with paraphrase judgments in each target language.
Like another work XNLI [4] on multilingual corpus creation,
the training set is from machine translate the original PAWS English training set (49,401 pairs).

We use PAWS-X to test the distillation performance in a multilingual setting.
The dataset are loaded using huggingface's API: https://huggingface.co/datasets/paws-x.
All Data will be automatically downloaded from HuggingFace (thus online connection is necessary) if it does not exist with the given dataset path.

### 1.3 Distillation Loss

The student model's loss is the weighted sum of hard_label_loss, soft_label_loss,
softmax_regression_loss, output_feature_loss, rkd_distance_loss, and rkd_angle_loss.

#### Output Feature Loss

Output Feature Loss is the distance (Euclidean/Cosine/...) between output feature
of teacher and student.

#### Softmax Regression Loss

[Knowledge Distillation via Softmax Regression Representation Learning](https://www.adrianbulat.com/downloads/ICLR2021/knowledge_distillation_via_softmax_regression_representation_learning.pdf) [5]

#### RKD Distance/Angle Loss

[Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068?context=cs.LG) [6]

### 1.4 Performance

#### GLUE

Here we show the importance of output feature loss.

```bash
glue_task=qnli
teacher_model=google/bert_uncased_L-12_H-768_A-12
student_model=google/bert_uncased_L-6_H-768_A-12
seed=123
max_epoch=12
metric="accuracy"
temperature=5
hard_label_weight=0.1
soft_label_weight=1

python3 automm_distillation_glue.py --teacher_model ${teacher_model} \
                                    --student_model ${student_model} \
                                    --seed ${seed} \
                                    --max_epoch ${max_epoch} \
                                    --hard_label_weight ${hard_label_weight} \
                                    --soft_label_weight ${soft_label_weight} \
                                    --glue_task ${glue_task}
```

| output_feature_loss_weight | Teacher Model Acc | Pretrained Model Acc | Student Model Acc | Distillation Ratio [2] | Speed Up |
| -------------------------- | ----------------- | -------------------- | ----------------- | ---------------------- | -------- |
| 0                          | 0.91726           | 0.89401              | 0.89713           | 0.13                   | 3.52x    |
| 0.01                       | 0.91726           | 0.89401              | 0.90298           | 0.39                   | 3.52x    |
| 0.1                        | 0.91726           | 0.89401              | 0.89365           | -0.02                  | 3.52x    |

#### PAWS-X

Here we show the importance of softmax regression loss and rkd loss. Some common settings:

```
pawsx_teacher_tasks = ["en", "de", "es", "fr", "ja", "ko", "zh"]
pawsx_student_tasks = ["en", "de", "es", "fr", "ja", "ko", "zh"]
Teacher Model = microsoft/mdeberta-v3-base
Student Model = nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large
seed = 123
precision = 16
max_epochs = 10
time_limit = None
num_gpu = -1
temperature = 5
hard_label_weight = 0.1
soft_label_weight = 1
softmax_regression_loss_type = mse
output_feature_loss_type = mse
```

To reproduce the best model, run
`python3 automm_distillation_pawsx.py --precision 16 --output_feature_loss_weight 0.01 --softmax_regression_weight 0.1 --rkd_distance_loss_weight 1 --rkd_angle_loss_weight 2`

|       Teacher Model        |                    Student Model                     | Softmax Regression Weight | RKD Distance Weight | RKD Angle Weight | Teacher Performance | No Distill Performance | Student Performance | Distillation Ratio | Speed Up |
| :------------------------: | :--------------------------------------------------: | :-----------------------: | :-----------------: | :--------------: | :-----------------: | :--------------------: | :-----------------: | :----------------: | :------: |
| microsoft/mdeberta-v3-base | nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large |             0             |          0          |        0         |       0.91293       |        0.88493         |       0.88857       |       0.1301       |  ~3.3x   |
| microsoft/mdeberta-v3-base | nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large |           0.01            |          0          |        0         |       0.91293       |        0.88493         |       0.8855        |      0.02041       |  ~3.3x   |
| microsoft/mdeberta-v3-base | nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large |            0.1            |          0          |        0         |       0.91293       |        0.88493         |       0.89093       |      0.21429       |  ~3.3x   |
| microsoft/mdeberta-v3-base | nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large |            0.1            |         0.1         |       0.2        |       0.91293       |        0.88493         |       0.89264       |      0.27551       |  ~3.3x   |
| microsoft/mdeberta-v3-base | nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large |            0.1            |          1          |        2         |       0.91293       |        0.88493         |        0.891        |      0.21684       |  ~3.3x   |
| microsoft/mdeberta-v3-base | nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large |            0.1            |         10          |        20        |       0.91293       |        0.88493         |       0.88971       |      0.17092       |  ~3.3x   |

### Reference

[1] Wang A, Singh A, Michael J, et al.
GLUE: A multi-task benchmark and analysis platform for natural language understanding[J].
arXiv preprint arXiv:1804.07461, 2018.

[2] H He, X Shi, J Mueller, S Zha, M Li, G Karypis
[Towards Automated Distillation: A Systematic Study of Knowledge Distillation in Natural Language Processing.](https://automl.cc/wp-content/uploads/2022/07/towards_automated_distillation.pdf)
International Conference on Automated Machine Learning: Late-Breaking Workshop, 2022

[3] Yang Y, Zhang Y, Tar C, et al.
PAWS-X: A cross-lingual adversarial dataset for paraphrase identification[J].
arXiv preprint arXiv:1908.11828, 2019.

[4] Conneau A, Lample G, Rinott R, et al.
XNLI: Evaluating cross-lingual sentence representations[J].
arXiv preprint arXiv:1809.05053, 2018.

[5] Yang J, Martinez B, Bulat A, et al. Knowledge distillation via softmax regression representation learning[C].
International Conference on Learning Representations (ICLR), 2021.

[6] Park W, Kim D, Lu Y, et al. Relational knowledge distillation[C].
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
2019: 3967-3976.
