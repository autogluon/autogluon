# AutoMMPredictor for Image, Text, and Tabular
:label:`sec_automm_customization`

This tutorial walks you through customizing various AutoMM configurations to give you advanced control of the fitting process. Specifically, AutoMM configurations consist of four parts: 
- optimization
- environment
- model
- data

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

### optimization.lr_schedule
Learning rate schedule.
- `"cosine_decay"`: the decay of learning rate follows the cosine curve.
- `"polynomial_decay"`: the learning rate is decayed based on polynomial functions. 
- `"linear_decay"`: linearly decay the learing rate.

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
By default, we disable `optimization.max_steps` by setting it as -1.
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
- `"greedy_soup"`: try to add the checkpoints from best to worst into the averaging pool and stop if the averaged checkpoint performance decreases. See [the paper](https://arxiv.org/pdf/2203.05482.pdf) for details.
- `"uniform_soup"`: average all the top k checkpoints as the final checkpoint.
- `"best"`: pick the checkpoint with the best validation performance.
```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.top_k_average_method": "greedy_soup"})
# average all the top k checkpoints
predictor.fit(hyperparameters={"optimization.top_k_average_method": "uniform_soup"})
```

### optimization.efficient_finetune
Finetune only a small portion of parameters instead of one whole pretrained backbone.
- `"bit_fit"`: bias parameters only.
- `"norm_fit"`: normalization parameters + bias parameters.
- `"lora"`: LoRA Adaptors. See [this paper](https://arxiv.org/abs/2106.09685) for details.
- `"lora_bias"`: LoRA Adaptors + bias parameters.
- `"lora_norm"`: LoRA Adaptors + normalization parameters + bias parameters.
```
# default used by AutoMM
predictor.fit(hyperparameters={"optimization.efficient_finetune": None})
# finetune only bias parameters
predictor.fit(hyperparameters={"optimization.efficient_finetune": "bit_fit"})
```

## Environment

### env.num_gpus
The number of gpus to use.
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
The batch size to use in each step of training. If `env.batch_size` is larger than `env.per_gpu_batch_size * env.num_gpus`, we accumulate gradients to reach the effective `env.batch_size` before performing one optimization step.
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
# use 2x per gpu batch size during prediction or evalution
predictor.fit(hyperparameters={"env.eval_batch_size_ratio": 2})
```

### env.precision
Support either double (`64`), float (`32`), bfloat16 (`"bf16"`), or half (`16`) precision training.

Half precision, or mixed precision, is the combined use of 32 and 16 bit floating points to reduce memory footprint during model training. This can result in improved performance, achieving +3X speedups on modern GPUs.
```
# default used by AutoMM
predictor.fit(hyperparameters={"env.precision": 16})
# use bfloat16
predictor.fit(hyperparameters={"env.precision": "bf16"})
```

### env.num_workers
The number of worker processes used by Pytorch dataloader in training.

```
# default used by AutoMM
predictor.fit(hyperparameters={"env.num_workers": 2})
# use 4 workers in the training dataloader
predictor.fit(hyperparameters={"env.num_workers": 4})
```

### env.num_workers_evaluation
The number of worker processes used by Pytorch dataloader in prediction or evaluation.

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
## Data
