# Customize AutoMM

:label:`sec_automm_customization`

AutoMM has a powerful yet easy-to-use configuration design.
This tutorial walks you through various AutoMM configurations to empower you the customization flexibility. Specifically, AutoMM configurations consist of several parts:

- optimization
- environment
- model
- data
- distiller

## Optimization

### optimization.learning_rate

Learning rate.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.learning_rate": 1.0e-4})
# set learning rate to 5.0e-4
predictor.fit(hyperparameters={"optimization.learning_rate": 5.0e-4})
```

### optimization.optim_type

Optimizer type.

- `"sgd"`: stochastic gradient descent with momentum.
- `"adam"`: a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments. See [this paper](https://arxiv.org/abs/1412.6980) for details.
- `"adamw"`: improves adam by decoupling the weight decay from the optimization step. See [this paper](https://arxiv.org/abs/1711.05101) for details.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.optim_type": "adamw"})
# use optimizer adam
predictor.fit(hyperparameters={"optimization.optim_type": "adam"})
```

### optimization.weight_decay

Weight decay.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.weight_decay": 1.0e-3})
# set weight decay to 1.0e-4
predictor.fit(hyperparameters={"optimization.weight_decay": 1.0e-4})
```

### optimization.lr_decay

Later layers can have larger learning rates than the earlier layers. The last/head layer
has the largest learning rate `optimization.learning_rate`. For a model with `n` layers, layer `i` has learning rate `optimization.learning_rate * optimization.lr_decay^(n-i)`. To use one uniform learning rate, simply set the learning rate decay to `1`.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.lr_decay": 0.9})
# turn off learning rate decay
predictor.fit(hyperparameters={"optimization.lr_decay": 1})
```

### optimization.lr_mult

While we are using two_stages lr choice,
The last/head layer has the largest learning rate `optimization.learning_rate` * `optimization.lr_mult`.
And other layers has normal learning rate `optimization.learning_rate`.
To use one uniform learning rate, simply set the learning rate multiple to `1`.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.lr_mult": 1})
# turn on two-stage lr for 10 times learning rate in head layer
predictor.fit(hyperparameters={"optimization.lr_mult": 10})
```

### optimization.lr_choice

We may want different layers to have different lr,
here we have strategy `two_stages` lr choice (see `optimization.lr_mult` section for more details),
or `layerwise_decay` lr choice (see `optimization.lr_decay` section for more details).
To use one uniform learning rate, simply set this to `""`.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.lr_choice": "layerwise_decay"})
# turn on two-stage lr choice
predictor.fit(hyperparameters={"optimization.lr_choice": "two_stages"})
```

### optimization.lr_schedule

Learning rate schedule.

- `"cosine_decay"`: the decay of learning rate follows the cosine curve.
- `"polynomial_decay"`: the learning rate is decayed based on polynomial functions.
- `"linear_decay"`: linearly decays the learing rate.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.lr_schedule": "cosine_decay"})
# use polynomial decay
predictor.fit(hyperparameters={"optimization.lr_schedule": "polynomial_decay"})
```

### optimization.max_epochs

Stop training once this number of epochs is reached.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.max_epochs": 10})
# train 20 epochs
predictor.fit(hyperparameters={"optimization.max_epochs": 20})
```

### optimization.max_steps

Stop training after this number of steps. Training will stop if `optimization.max_steps` or `optimization.max_epochs` have reached (earliest).
By default, we disable `optimization.max_steps` by setting it to -1.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.max_steps": -1})
# train 100 steps
predictor.fit(hyperparameters={"optimization.max_steps": 100})
```

### optimization.warmup_steps

Warm up the learning rate from 0 to `optimization.learning_rate` within this percentage of steps at the beginning of training.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.warmup_steps": 0.1})
# do learning rate warmup in the first 20% steps.
predictor.fit(hyperparameters={"optimization.warmup_steps": 0.2})
```

### optimization.patience

Stop training after this number of checks with no improvement. The check frequency is controlled by `optimization.val_check_interval`.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.patience": 10})
# set patience to 5 checks
predictor.fit(hyperparameters={"optimization.patience": 5})
```

### optimization.val_check_interval

How often within one training epoch to check the validation set. Can specify as float or int.

- pass a float in the range [0.0, 1.0] to check after a fraction of the training epoch.
- pass an int to check after a fixed number of training batches.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.val_check_interval": 0.5})
# check validation set 4 times during a training epoch
predictor.fit(hyperparameters={"optimization.val_check_interval": 0.25})
```

### optimization.gradient_clip_algorithm

The gradient clipping algorithm to use. Support to clip gradients by value or norm.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.gradient_clip_algorithm": "norm"})
# clip gradients by value
predictor.fit(hyperparameters={"optimization.gradient_clip_algorithm": "value"})
```

### optimization.gradient_clip_val

Gradient clipping value, which can be the absolute value or gradient norm depending on the choice of `optimization.gradient_clip_algorithm`.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.gradient_clip_val": 1})
# cap the gradients to 5
predictor.fit(hyperparameters={"optimization.gradient_clip_val": 5})
```

### optimization.track_grad_norm

Track the p-norm of gradients during training. May be set to ‘inf’ infinity-norm. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before logging them.

```
# default used by AutoMM (no tracking)
predictor.fit(hyperparameters={"optimization.track_grad_norm": -1})
# track the 2-norm
predictor.fit(hyperparameters={"optimization.track_grad_norm": 2})
```

### optimization.log_every_n_steps

How often to log within steps.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.log_every_n_steps": 10})
# log once every 50 steps
predictor.fit(hyperparameters={"optimization.log_every_n_steps": 50})
```

### optimization.top_k

Based on the validation score, choose top k model checkpoints to do model averaging.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.top_k": 3})
# use top 5 checkpoints
predictor.fit(hyperparameters={"optimization.top_k": 5})
```

### optimization.top_k_average_method

Use what strategy to average the top k model checkpoints.

- `"greedy_soup"`: tries to add the checkpoints from best to worst into the averaging pool and stop if the averaged checkpoint performance decreases. See [the paper](https://arxiv.org/pdf/2203.05482.pdf) for details.
- `"uniform_soup"`: averages all the top k checkpoints as the final checkpoint.
- `"best"`: picks the checkpoint with the best validation performance.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.top_k_average_method": "greedy_soup"})
# average all the top k checkpoints
predictor.fit(hyperparameters={"optimization.top_k_average_method": "uniform_soup"})
```

### optimization.efficient_finetune

Options for parameter-efficient finetuning. Parameter-efficient finetuning means to finetune only a small portion of parameters instead of the whole pretrained backbone.

- `"bit_fit"`: bias parameters only. See [this paper](https://arxiv.org/pdf/2106.10199.pdf) for details.
- `"norm_fit"`: normalization parameters + bias parameters. See [this paper](https://arxiv.org/pdf/2003.00152.pdf) for details.
- `"lora"`: LoRA Adaptors. See [this paper](https://arxiv.org/pdf/2106.09685.pdf) for details.
- `"lora_bias"`: LoRA Adaptors + bias parameters.
- `"lora_norm"`: LoRA Adaptors + normalization parameters + bias parameters.
- `"ia3"`: IA3 algorithm. See [this paper](https://arxiv.org/abs/2205.05638) for details.
- `"ia3_bias"`: IA3 + bias parameters.
- `"ia3_norm"`: IA3 + normalization parameters + bias parameters.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.efficient_finetune": None})
# finetune only bias parameters
predictor.fit(hyperparameters={"optimization.efficient_finetune": "bit_fit"})
# finetune with IA3 + BitFit
predictor.fit(hyperparameters={"optimization.efficient_finetune": "ia3_bias"})
```

### optimization.skip_final_val

Whether to skip the final validation after training is signaled to stop.

```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.skip_final_val": False})
# skip the final validation
predictor.fit(hyperparameters={"optimization.top_k_average_method": True})
```

## Environment

### env.num_gpus

The number of gpus to use. If given -1, we count the GPUs by `env.num_gpus = torch.cuda.device_count()`.

```
# by default, all available gpus are used by AutoMM
predictor.fit(hyperparameters={"env.num_gpus": -1})
# use 1 gpu only
predictor.fit(hyperparameters={"env.num_gpus": 1})
```

### env.per_gpu_batch_size

The batch size for each GPU.

```
# default used by AutoMM
predictor.fit(hyperparameters={"env.per_gpu_batch_size": 8})
# use batch size 16 per GPU
predictor.fit(hyperparameters={"env.per_gpu_batch_size": 16})
```

### env.batch_size

The batch size to use in each step of training. If `env.batch_size` is larger than `env.per_gpu_batch_size * env.num_gpus`, we accumulate gradients to reach the effective `env.batch_size` before performing one optimization step. The accumulation steps are calculated by `env.batch_size // (env.per_gpu_batch_size * env.num_gpus)`.

```
# default used by AutoMM
predictor.fit(hyperparameters={"env.batch_size": 128})
# use batch size 256
predictor.fit(hyperparameters={"env.batch_size": 256})
```

### env.eval_batch_size_ratio

Prediction or evaluation uses a larger per gpu batch size `env.per_gpu_batch_size * env.eval_batch_size_ratio`.

```
# default used by AutoMM
predictor.fit(hyperparameters={"env.eval_batch_size_ratio": 4})
# use 2x per gpu batch size during prediction or evaluation
predictor.fit(hyperparameters={"env.eval_batch_size_ratio": 2})
```

### env.precision

Support either double (`64`), float (`32`), bfloat16 (`"bf16"`), or half (`16`) precision training.

Half precision, or mixed precision, is the combined use of 32 and 16 bit floating points to reduce memory footprint during model training. This can result in improved performance, achieving +3x speedups on modern GPUs.

```
# default used by AutoMM
predictor.fit(hyperparameters={"env.precision": 16})
# use bfloat16
predictor.fit(hyperparameters={"env.precision": "bf16"})
```

### env.num_workers

The number of worker processes used by the Pytorch dataloader in training. Note that more workers don't always bring speedup especially when `env.strategy = "ddp_spawn"`.
For more details, see the guideline [here](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu.html#distributed-data-parallel).

```
# default used by AutoMM
predictor.fit(hyperparameters={"env.num_workers": 2})
# use 4 workers in the training dataloader
predictor.fit(hyperparameters={"env.num_workers": 4})
```

### env.auto_select_gpus

If enabled and devices is an integer, pick available GPUs automatically. This is especially useful when GPUs are configured to be in “exclusive mode”, such that only one process at a time can access them.
For more details, see the guideline [here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).

```
# default used by AutoMM
predictor.fit(hyperparameters={"env.auto_select_gpus": True})
# disable auto select gpus
predictor.fit(hyperparameters={"env.auto_select_gpus": False})
```

### env.num_workers_evaluation

The number of worker processes used by the Pytorch dataloader in prediction or evaluation.

```
# default used by AutoMM
predictor.fit(hyperparameters={"env.num_workers_evaluation": 2})
# use 4 workers in the prediction/evaluation dataloader
predictor.fit(hyperparameters={"env.num_workers_evaluation": 4})
```

### env.strategy

Distributed training mode.

- `"dp"`: data parallel.
- `"ddp"`: distributed data parallel (python script based).
- `"ddp_spawn"`: distributed data parallel (spawn based).

See [here](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu.html#distributed-modes) for more details.

```
# default used by AutoMM
predictor.fit(hyperparameters={"env.strategy": "ddp_spawn"})
# use ddp during training
predictor.fit(hyperparameters={"env.strategy": "ddp"})
```

## Model

### model.names

Choose what types of models to use.

- `"hf_text"`: the pretrained text models from [Huggingface](https://huggingface.co/).
- `"timm_image"`: the pretrained image models from [TIMM](https://github.com/rwightman/pytorch-image-models/tree/master/timm/models).
- `"clip"`: the pretrained CLIP models.
- `"categorical_mlp"`: MLP for categorical data.
- `"numerical_mlp"`: MLP for numerical data.
- `"categorical_transformer"`: [FT-Transformer](https://arxiv.org/pdf/2106.11959.pdf) for categorical data.
- `"numerical_transformer"`: [FT-Transformer](https://arxiv.org/pdf/2106.11959.pdf) for numerical data.
- `"fusion_mlp"`: MLP-based fusion for features from multiple backbones.
- `"fusion_transformer"`: transformer-based fusion for features from multiple backbones.

If no data of one modality is detected, the related model types will be automatically removed in training.

```
# default used by AutoMM
predictor.fit(hyperparameters={"model.names": ["hf_text", "timm_image", "clip", "categorical_mlp", "numerical_mlp", "fusion_mlp"]})
# use only text models
predictor.fit(hyperparameters={"model.names": ["hf_text"]})
# use only image models
predictor.fit(hyperparameters={"model.names": ["timm_image"]})
# use only clip models
predictor.fit(hyperparameters={"model.names": ["clip"]})
```

### model.hf_text.checkpoint_name

Specify a text backbone supported by the Hugginface [AutoModel](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodel).

```
# default used by AutoMM
predictor.fit(hyperparameters={"model.hf_text.checkpoint_name": "google/electra-base-discriminator"})
# choose roberta base
predictor.fit(hyperparameters={"model.hf_text.checkpoint_name": "roberta-base"})
```

### model.hf_text.pooling_mode

The feature pooling mode for transformer architectures.

- `cls`: uses the cls feature vector to represent a sentence.
- `mean`: averages all the token feature vectors to represent a sentence.

```
# default used by AutoMM
predictor.fit(hyperparameters={"model.hf_text.pooling_mode": "cls"})
# using the mean pooling
predictor.fit(hyperparameters={"model.hf_text.pooling_mode": "mean"})
```

### model.hf_text.tokenizer_name

Choose the text tokenizer. It is recommended to use the default auto tokenizer.

- `hf_auto`: the [Huggingface auto tokenizer](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer).
- `bert`: the [BERT tokenizer](https://huggingface.co/docs/transformers/v4.21.1/en/model_doc/bert#transformers.BertTokenizer).
- `electra`: the [ELECTRA tokenizer](https://huggingface.co/docs/transformers/v4.21.1/en/model_doc/electra#transformers.ElectraTokenizer).
- `clip`: the [CLIP tokenizer](https://huggingface.co/docs/transformers/v4.21.1/en/model_doc/clip#transformers.CLIPTokenizer).

```
# default used by AutoMM
predictor.fit(hyperparameters={"model.hf_text.tokenizer_name": "hf_auto"})
# using the tokenizer of the ELECTRA model
predictor.fit(hyperparameters={"model.hf_text.tokenizer_name": "electra"})
```

### model.hf_text.max_text_len

Set the maximum text length. Different models may allow different maximum lengths. If `model.hf_text.max_text_len` > 0, we choose the minimum between `model.hf_text.max_text_len` and the maximum length allowed by the model. Setting `model.hf_text.max_text_len` <= 0 would use the model's maximum length.

```
# default used by AutoMM
predictor.fit(hyperparameters={"model.hf_text.max_text_len": 512})
# set to use the length allowed by the tokenizer.
predictor.fit(hyperparameters={"model.hf_text.max_text_len": -1})
```

### model.hf_text.insert_sep

Whether to insert the SEP token between texts from different columns of a dataframe.

```
# default used by AutoMM
predictor.fit(hyperparameters={"model.hf_text.insert_sep": True})
# use no SEP token.
predictor.fit(hyperparameters={"model.hf_text.insert_sep": False})
```

### model.hf_text.text_segment_num

How many text segments are used in a token sequence. Each text segment has one [token type ID](https://huggingface.co/transformers/v2.11.0/glossary.html#token-type-ids). We choose the minimum between `model.hf_text.text_segment_num` and the default used by the model.

```
# default used by AutoMM
predictor.fit(hyperparameters={"model.hf_text.text_segment_num": 2})
# use 1 text segment
predictor.fit(hyperparameters={"model.hf_text.text_segment_num": 1})
```

### model.hf_text.stochastic_chunk

Whether to randomly cut a text chunk if a sample's text token number is larger than `model.hf_text.max_text_len`. If False, cut a token sequence from index 0 to the maximum allowed length. Otherwise, randomly sample a start index to cut a text chunk.

```
# default used by AutoMM
predictor.fit(hyperparameters={"model.hf_text.stochastic_chunk": False})
# select a stochastic text chunk if a text sequence is over-long
predictor.fit(hyperparameters={"model.hf_text.stochastic_chunk": True})
```

### model.hf_text.text_aug_detect_length

Perform text augmentation only when the text token number is no less than `model.hf_text.text_aug_detect_length`.

```
# default used by AutoMM
predictor.fit(hyperparameters={"model.hf_text.text_aug_detect_length": 10})
# Allow text augmentation for texts whose token number is no less than 5
predictor.fit(hyperparameters={"model.hf_text.text_aug_detect_length": 5})
```

### model.hf_text.text_trivial_aug_maxscale

Set the maximum percentage of text tokens to conduct data augmentation. For each text token sequence, we randomly sample a percentage in [0, `model.hf_text.text_trivial_aug_maxscale`] and one operation from four trivial augmentations, including synonym replacement, random word swap, random word deletion, and random punctuation insertion, to do text augmentation.

```
# by default, AutoMM doesn't do text augmentation
predictor.fit(hyperparameters={"model.hf_text.text_trivial_aug_maxscale": 0})
# Enable trivial augmentation by setting the max scale to 0.1
predictor.fit(hyperparameters={"model.hf_text.text_trivial_aug_maxscale": 0.1})
```

### model.hf_text.gradient_checkpointing

Whether to turn on gradient checkpointing to reduce the memory consumption for calculating gradients. For more about gradient checkpointing, feel free to refer to [relevant tutorials](https://github.com/cybertronai/gradient-checkpointing).

```
# by default, AutoMM doesn't turn on gradient checkpointing
predictor.fit(hyperparameters={"model.hf_text.gradient_checkpointing": False})
# Turn on gradient checkpointing
predictor.fit(hyperparameters={"model.hf_text.gradient_checkpointing": True})
```

### model.timm_image.checkpoint_name

Select an image backbone from [TIMM](https://github.com/rwightman/pytorch-image-models/tree/master/timm/models).

```
# default used by AutoMM
predictor.fit(hyperparameters={"model.timm_image.checkpoint_name": "swin_base_patch4_window7_224"})
# choose a vit base
predictor.fit(hyperparameters={"model.timm_image.checkpoint_name": "vit_base_patch32_224"})
```

## Data

### data.image.missing_value_strategy

How to deal with missing images, opening which fails.

- `"skip"`: skip a sample with missing images.
- `"zero"`: use zero image to replace a missing image.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.image.missing_value_strategy": "zero"})
# skip the image
predictor.fit(hyperparameters={"data.image.missing_value_strategy": "skip"})
```

### data.text.normalize_text
Whether to normalize text with encoding problems. If True, TextProcessor will run through a series of encoding and decoding for text normalization. Please refer to the [Example](https://github.com/autogluon/autogluon/tree/master/examples/automm/kaggle_feedback_prize) of Kaggle competition for applying text normalization.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.text.normalize_text": False})
# turn on text normalization
predictor.fit(hyperparameters={"data.text.normalize_text": True})
```

### data.categorical.convert_to_text

Whether to treat categorical data as text. If True, no categorical models, e.g., `"categorical_mlp"` and `"categorical_transformer"`, would be used.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.categorical.convert_to_text": True})
# turn off the conversion
predictor.fit(hyperparameters={"data.categorical.convert_to_text": False})
```

### data.numerical.convert_to_text

Whether to convert numerical data to text. If True, no numerical models e.g., `"numerical_mlp"` and `"numerical_transformer"`, would be used.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.numerical.convert_to_text": False})
# turn on the conversion
predictor.fit(hyperparameters={"data.numerical.convert_to_text": True})
```

### data.numerical.scaler_with_mean

If True, center the numerical data (not including the numerical labels) before scaling.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.numerical.scaler_with_mean": True})
# turn off centering
predictor.fit(hyperparameters={"data.numerical.scaler_with_mean": False})
```

### data.numerical.scaler_with_std

If True, scale the numerical data (not including the numerical labels) to unit variance.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.numerical.scaler_with_std": True})
# turn off scaling
predictor.fit(hyperparameters={"data.numerical.scaler_with_std": False})
```

### data.label.numerical_label_preprocessing

How to process the numerical labels in regression tasks.

- `"standardscaler"`: standardizes numerical labels by removing the mean and scaling to unit variance.
- `"minmaxscaler"`: transforms numerical labels by scaling each feature to range (0, 1).

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.label.numerical_label_preprocessing": "standardscaler"})
# scale numerical labels to (0, 1)
predictor.fit(hyperparameters={"data.label.numerical_label_preprocessing": "minmaxscaler"})
```

### data.pos_label

The positive label in a binary classification task. Users need to specify this label to properly use some metrics, e.g., roc_auc, average_precision, and f1.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.pos_label": None})
# assume the labels are ["changed", "not changed"] and "changed" is the positive label
predictor.fit(hyperparameters={"data.pos_label": "changed"})
```

### data.mixup.turn_on

If True, use Mixup in training.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.mixup.turn_on": False})
# turn on Mixup
predictor.fit(hyperparameters={"data.mixup.turn_on": True})
```

### data.mixup.mixup_alpha

Mixup alpha value. Mixup is active if `data.mixup.mixup_alpha` > 0.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.mixup.mixup_alpha": 0.8})
# set it to 1.0 to turn off Mixup
predictor.fit(hyperparameters={"data.mixup.mixup_alpha": 1.0})
```

### data.mixup.cutmix_alpha

Cutmix alpha value. Cutmix is active if `data.mixup.cutmix_alpha` > 0.

```
# by default, Cutmix is turned off by using alpha 1.0
predictor.fit(hyperparameters={"data.mixup.cutmix_alpha": 1.0})
# turn it on by choosing a number in range (0, 1)
predictor.fit(hyperparameters={"data.mixup.cutmix_alpha": 0.8})
```

### data.mixup.prob

The probability of conducting Mixup or Cutmix if enabled.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.mixup.prob": 1.0})
# set probability to 0.5
predictor.fit(hyperparameters={"data.mixup.prob": 0.5})
```

### data.mixup.switch_prob

The probability of switching to Cutmix instead of Mixup when both are active.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.mixup.switch_prob": 0.5})
# set probability to 0.7
predictor.fit(hyperparameters={"data.mixup.switch_prob": 0.7})
```

### data.mixup.mode

How to apply Mixup or Cutmix params (per `"batch"`, `"pair"` (pair of elements), `"elem"` (element)).
See [here](https://github.com/rwightman/pytorch-image-models/blob/d30685c283137b4b91ea43c4e595c964cd2cb6f0/timm/data/mixup.py#L211-L216) for more details.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.mixup.mode": "batch"})
# use "pair"
predictor.fit(hyperparameters={"data.mixup.mode": "pair"})
```

### data.mixup.label_smoothing

Apply label smoothing to the mixed label tensors.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.mixup.label_smoothing": 0.1})
# set it to 0.2
predictor.fit(hyperparameters={"data.mixup.label_smoothing": 0.2})
```

### data.mixup.turn_off_epoch

Stop Mixup or Cutmix after reaching this number of epochs.

```
# default used by AutoMM
predictor.fit(hyperparameters={"data.mixup.turn_off_epoch": 5})
# turn off mixup after 7 epochs
predictor.fit(hyperparameters={"data.mixup.turn_off_epoch": 7})
```

## Distiller

### distiller.soft_label_loss_type

What loss to compute when using teacher's output (logits) to supervise student's.

```
# default used by AutoMM for classification
predictor.fit(hyperparameters={"distiller.soft_label_loss_type": "cross_entropy"})
# default used by AutoMM for regression
predictor.fit(hyperparameters={"distiller.soft_label_loss_type": "mse"})
```

### distiller.temperature

Before computing the soft label loss, scale the teacher and student logits with it (teacher_logits / temperature, student_logits / temperature).

```
# default used by AutoMM for classification
predictor.fit(hyperparameters={"distiller.temperature": 5})
# set temperature to 1
predictor.fit(hyperparameters={"distiller.temperature": 1})
```

### distiller.hard_label_weight

Scale the student's hard label (groundtruth) loss with this weight (hard_label_loss \* hard_label_weight).

```
# default used by AutoMM for classification
predictor.fit(hyperparameters={"distiller.hard_label_weight": 0.2})
# set not to scale the hard label loss
predictor.fit(hyperparameters={"distiller.hard_label_weight": 1})
```

### distiller.soft_label_weight

Scale the student's soft label (teacher's output) loss with this weight (soft_label_loss \* soft_label_weight).

```
# default used by AutoMM for classification
predictor.fit(hyperparameters={"distiller.soft_label_weight": 50})
# set not to scale the soft label loss
predictor.fit(hyperparameters={"distiller.soft_label_weight": 1})
```

Please bear with us while we analyse the content
