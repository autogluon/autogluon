# Advanced Topics

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Single GPU Billion-scale Model Training via Parameter-Efficient Finetuning
  :link: efficient_finetuning_basic.html

  How to take advantage of large foundation models with the help of parameter-efficient finetuning.
  In the tutorial, we will use combine IA^3, BitFit, and gradient checkpointing to finetune FLAN-T5-XL.
:::

:::{grid-item-card} Hyperparameter Optimization in AutoMM
  :link: hyperparameter_optimization.html

  How to do hyperparameter optimization in AutoMM.
:::

:::{grid-item-card} Knowledge Distillation in AutoMM
  :link: model_distillation.html

  How to do knowledge distillation in AutoMM.
:::

:::{grid-item-card} Customize AutoMM
  :link: customization.html

  How to customize AutoMM configurations.
:::

:::{grid-item-card} AutoMM Presets
  :link: presets.html

  How to use AutoMM presets.
:::

:::{grid-item-card} Few Shot Learning with AutoMM
  :link: few_shot_learning.html

  How to use foundation models + SVM for few shot learning.
:::

:::{grid-item-card} Handling Class Imbalance with AutoMM - Focal Loss
  :link: focal_loss.html

  How to use AutoMM to handle class imbalance.
:::

:::{grid-item-card} Faster Prediction with TensorRT
  :link: tensorrt.html

  How to use TensorRT in accelerating AutoMM model inference.
:::

:::{grid-item-card} Continuous Training with AutoMM
  :link: continuous_training.html

  Different use cases for continuous training with AutoMM.
:::

:::{grid-item-card} AutoMM Problem Types and Evaluation Metrics.
  :link: problem_types_and_metrics.html

  A comprehensive guide to AutoGluon's supported problem types and their evaluation metrics.
:::

:::{grid-item-card} Multiple Label Columns
  :link: multiple_label_columns.html

  How to handle multiple label columns with AutoGluon MultiModal.
:::

::::

```{toctree}
---
maxdepth: 1
hidden: true
---

problem_types_and_metrics
hyperparameter_optimization
continuous_training
customization
model_distillation
efficient_finetuning_basic
few_shot_learning
focal_loss
presets
tensorrt
multiple_label_columns
```
