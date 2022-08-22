# Model Distillation in AutoMM

Examples showing how to use `MultiModalPredictor` for model distillation.

## 1. Distillation on GLUE tasks

### 1.1 Example
[`automm_distillation.py`](./automm_distillation.py) : This example provides a use case 

To run the example:

```python automm_distillation.py --glue_task stsb --<other_flag> <value>```
   - `glue_task` determines to run the experiments on which GLUE task, refers to [Dataset Section](###1.2-Datasets).
   - `teacher_model` is the name of teacher model
   - `student_model` is the name of the student model. It is recommended to have the same backbone as teacher model (e.g. both ELECTRA or both BERT).
   - `seed` determines the random seed. Default is 123.
   - `max_epochs` determines the max epoch for training student model.
   - `time_limit` determines the max time to train each model. The unit is second.
   - `num_gpu` determines num of gpu used to train the models. Default is -1 which means using all.
   - `temperature` determines the temperature for soft cross entropy.
   - `hard_label_weight` determines the weight of hard label loss (ground truth logits).
   - `soft_label_weight` determines the weight of soft label loess (teacher model's output).
   - `output_feature_loss_weight` determines the weight of output_feature loss
   - `output_feature_loss_type` determines the loss function of output_feature loss, currently support "mean_square_error" and "cosine_distance"
   - `finetuned_model_cache_folder` is the path to cache models trained without distillation.
   -  `retrain` determines if the models without distillation should be retrained or load from cache.

### 1.2 Dataset
We borrow 7 NLP tasks in GLUE [1], and use identically the same abbreviation as in [1] to name each tasks,
i.e. "mnli(m/mm)", "qqp", "qnli", "sst2", "stsb", "mrpc", and "rte".
The dataset are loaded using huggingface's API: https://huggingface.co/datasets/glue.
All Data will be automatically downloaded from HuggingFace (thus online connection is necessary) if it does not exist with the given dataset path.

In GLUE, labels in test sets are private and thus we use training set for training/validation and use validation set for testing.

### 1.3 Distillation Loss
The student model's loss is defined by 
```
    def _compute_hard_label_loss(
        self,
        output: dict,
        label: torch.Tensor,
    ):
        loss = 0
        for per_output in output.values():
            weight = per_output[WEIGHT] if WEIGHT in per_output else 1
            loss += (
                self.hard_label_loss_func(
                    input=per_output[LOGITS].squeeze(dim=1),
                    target=label,
                )
                * weight
            )

        return loss

    def _compute_soft_label_loss(
        self,
        student_output: dict,
        teacher_output: dict,
    ):
        student_logits = student_output[self.student_model.prefix][LOGITS].squeeze(dim=1)
        soft_labels = teacher_output[self.teacher_model.prefix][LOGITS].squeeze(dim=1)
        student_logits = student_logits / self.temperature
        soft_labels = soft_labels / self.temperature

        if isinstance(self.soft_label_loss_func, nn.CrossEntropyLoss):
            soft_labels = F.softmax(soft_labels, dim=-1)

        loss = self.soft_label_loss_func(
            input=student_logits,
            target=soft_labels,
        )
        return loss

    def _compute_output_feature_loss(
        self,
        student_output: dict,
        teacher_output: dict,
    ):
        student_result = student_output[self.student_model.prefix][FEATURES].squeeze(dim=1)
        teacher_result = teacher_output[self.teacher_model.prefix][FEATURES].squeeze(dim=1)

        loss = self.output_feature_loss_func(
            input=student_result,
            target=teacher_result,
        )
        return loss

    def _compute_loss(
        self,
        student_output: dict,
        teacher_output: dict,
        label: torch.Tensor,
    ):
        loss = 0
        hard_label_loss = self._compute_hard_label_loss(
            output=student_output,
            label=label,
        )
        loss += hard_label_loss * self.hard_label_weight

        soft_label_loss = self._compute_soft_label_loss(
            student_output=student_output,
            teacher_output=teacher_output,
        )
        loss += soft_label_loss * self.soft_label_weight

        output_feature_loss = self._compute_output_feature_loss(
            student_output=student_output,
            teacher_output=teacher_output,
        )
        loss += output_feature_loss * self.output_feature_loss_weight

        return loss
```

### 1.4 Performance
```
glue_task = qnli
Teacher Model = google/bert_uncased_L-12_H-768_A-12
Student Model = google/bert_uncased_L-6_H-768_A-12
seed = 123
max_epoch = 12
metric = "accuracy"
temperature = 5
hard_label_weight = 0.1
soft_label_weight = 1
```

`python3 automm_distillation.py --teacher_model google/bert_uncased_L-12_H-768_A-12 --student_model google/bert_uncased_L-6_H-768_A-12 --seed 123 --max_epoch 8 --hard_label_weight 0.5 --soft_label_weight 5`

output_feature_loss_weight | Teacher Model Acc | Pretrained Model Acc | Student Model Acc | Distillation Ratio [2] | Speed Up 
----------------------|-------------------|----------------------|-------------------|------------------------|----
0                     | 0.91726           | 0.89401              | 0.89713           | 0.13                   | 3.52x
0.01                  | 0.91726           | 0.89401              | 0.90298           | 0.39                   | 3.52x
0.1                   | 0.91726           | 0.89401              | 0.89365           | -0.02                  | 3.52x

### 1.5 TODOs for Distillation
- Find a better hyperparameter setting (or autotuning) for distillation.

### Reference
[1] Wang A, Singh A, Michael J, et al. 
GLUE: A multi-task benchmark and analysis platform for natural language understanding[J]. 
arXiv preprint arXiv:1804.07461, 2018.

[2] H He, X Shi, J Mueller, S Zha, M Li, G Karypis
[Towards Automated Distillation: A Systematic Study of Knowledge Distillation in Natural Language Processing.](https://automl.cc/wp-content/uploads/2022/07/towards_automated_distillation.pdf)
International Conference on Automated Machine Learning: Late-Breaking Workshop, 2022 
