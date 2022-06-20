Examples showing how to use `AutoMMPredictor` for model distillation.

## 1. Distillation on GLUE tasks

### 1.1 Example
[`automm_distillation.py`](./automm_distillation.py) : This example provides a use case 
for using distillation in AutoMMPredictor, as well as the evaluation of the distillation result.

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
   - `hard_label_weight` determines the weight of hard labels (ground truth logits).
   - `soft_label_weight` determines the weight of soft labels (teacher model's output).

### 1.2 Dataset
We borrow 7 NLP tasks in GLUE [1], and use identically the same abbreviation as in [1] to name each tasks,
i.e. "mnli(m/mm)", "qqp", "qnli", "sst2", "stsb", "mrpc", and "rte".
The dataset are loaded using huggingface's API: https://huggingface.co/datasets/glue.
All Data will be automatically downloaded from HuggingFace (thus online connection is necessary) if it does not exist with the given dataset path. 

In GLUE, labels in test sets are private and thus we use training set for training/validation and use validation set for testing.

### 1.3 Distillation Loss
The student model's loss is defined by 
```
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
```

### 1.4 Performance
```
glue_task = qnli
Teacher Model = google/bert_uncased_L-12_H-768_A-12
Student Model = google/bert_uncased_L-6_H-768_A-12
seed = 123
max_epoch = 8
time_limit = 3600
metric = "accuracy"
```

`python3 automm_distillation.py --teacher_model google/bert_uncased_L-12_H-768_A-12 --student_model google/bert_uncased_L-6_H-768_A-12 --seed 123 --max_epoch 8 --hard_label_weight 0.5 --soft_label_weight 5`

temperature | hard_label_weight | soft_label_weight | Teacher Model Acc | Pretrained Model Acc | Student Model Acc | Distillation Ratio | Speed Up 
------------|-------------------|-------------------|-------------------|----------------------|-------------------|--------------------|----
5           | 0.2               | 50                | 0.91067           | 0.89420              | 0.90244           | 0.50               | 1.08x
3           | 0.4               | 10                | 0.91067           | 0.89420              | 0.90189           | 0.47               | 1.08x
5           | 0.5               | 5                 | 0.91067           | 0.89420              | 0.90646           | 0.74               | 1.08x

### 1.5 TODOs for Distillation
- Add intermediate distillation.
- Find a better hyperparameter setting (or autotuning) for distillation.

### Reference
[1] Wang A, Singh A, Michael J, et al. 
GLUE: A multi-task benchmark and analysis platform for natural language understanding[J]. 
arXiv preprint arXiv:1804.07461, 2018.