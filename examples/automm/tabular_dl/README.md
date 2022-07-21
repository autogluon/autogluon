# Advanced models in autogluon.multimodal for tabular data


### 1. Run Example
[`example_tabular.py`](./example_tabular.py) : This example provides a use case for the pure *tabular* data, including pure numerical features and numerical + categorical feartures, with FT_Transformer [1].

To run the example: 

```python example_tabular.py --dataset_name ad --dataset_dir ./dataset --exp_dir ./result```
   - `dataset_name` determines which dataset to run the experinments, refers to [Dataset Section](###2.-Datasets).
   - `dataset_dir` is the path to the dataset(s). If the datasets do not present in this path, it will be automatically downloaded.
   - `exp_dir` is the output path to store the weights and loggings. 
   - `gpu_id` specifies the GPU to use (optional).
   - `seed` determines the random seed (optional). Default is 0.
   - `lr` specifies the inital learning rate (optional). Default is `1e-04`.
   - `end_lr` specifies the end learning rate (optional). Default is `1e-04`.


### 2. Datasets
We borrowed 11 tabular datasets provided by [1], and use identically the same abbreviation as Table 1 in the original paper [1] to name each datasets. 
The datasets provided by https://github.com/Yura52/tabular-dl-revisiting-models are all in  `Numpy.darray` format (can be downloaded from https://www.dropbox.com/s/o53umyg6mn3zhxy/data.tar.gz?dl=1). 
These Data in `Numpy.ndarray` was first pre-processedin into `.csv` format, which can be loaded by `pandas.Dataframe` as the input to `MultiModalPredictor`. 
All Data can be automatically downloaded from s3 (online connection is necessary) if it does not exisit with the given dataset path `dataset_dir`. 


### 3. FT-Transformer
We categorize the original FT_Transformer to two models in `MultiModalPredictor`, namely `numerical_transformer` and `categorical transformer`, which depends on the modaility of input tabular data (i.e., numerical v.s. categorical). The two models share most of the common features:
   - `out_features` is the output feature size.
   - `d_token` is the dimension of the embedding tokens.
   - `num_trans_blocks` is the number of transformer blocks.
   - `num_attn_heads` is the number of the attention heads in multi-headed self attention layer in each transformer block.
   - `ffn_dropout` is the dropout rate in feadforward layer.
   - `ffn_activation` determines the activation fuction in feadforward layer. We support `relu`, `gelu`, `reglu` and `leaky_relu`.
   - `attention_dropout` is the dropout rate in attention layer.
`numerical_transformer` supports an additional feature:
   - `embedding_arch` is a list containing the names of embedding layers as described in [2]. Whatever the embedding layers are selected, the shape of the output embedding is `batch_size * number_of_numerical_features * d_token`.
  
These features can be tuned using `hyperparameters` in `MultiModalPredictor. For example: 
```python
hyperparameters = {
   "model.names": ["categorical_transformer","numerical_transformer","fusion_transformer"],
   "model.categorical_transformer.num_trans_blocks": 1,
   "ffn_dropout": 0.0,
}
```


### 4. Results

Datasets | ca | ad | he | ja | hi | al | ep | ye | co | ya | mi 
----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  
metrics | rmse | acc | acc | acc | acc | acc | acc | rmse | acc | rmse | rmse
problem_type | regression | binary | multiclass | multiclass | binary | multiclass | binary | regression | multiclass | regression | regression
#objects | 20640 | 48842 | 65196 | 83733 | 98050 | 108000 | 500000 | 515345 | 581012 | 709877 | 1200192
#num. | 8 | 6 | 27 | 54 | 28 | 128 | 2000 | 90 | 54 | 699 | 136
#cat. | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
#classes | - | 2 | 100 | 4 | 2 | 1000 | 2 | - | 7 | - | -
Best in [1] | 0.459 | 0.859 | 0.396 | 0.732 | 0.729 | 0.963 | 0.8982 | 8.794 | 0.970 | 0.753 | 0.745
FT-Transformer in [1] | 0.459 | 0.859 | 0.391 | 0.732 | 0.729 | 0.960 | 0.8982 | 8.855 | 0.970 | 0.756 | 0.746
AutoMM FT-Transformer | 0.482 | 0.859 | 0.379 | 0.721 | 0.726 | 0.949 | RuntimeError | 8.891 | 0.963 | 0.769 | 0.761

`FT-Transformer in [1]` row leverages parameters searching, and `AutoMM FT-Transformer` row use a fixed training configurations.

You can reproduce the `AutoMM FT-Transformer` row by running:
```bash
bash run_all.sh
```
with overrideng the following hyperparameters:
```python
automm_hyperparameters = {
    "data.categorical.convert_to_text": False,
    "model.names": ["categorical_transformer", "numerical_transformer", "fusion_transformer"],
    "model.numerical_transformer.embedding_arch": ["linear"],
    "env.batch_size": 128,
    "env.per_gpu_batch_size": 128,
    "env.eval_batch_size_ratio": 1,
    "env.num_workers": 12,
    "env.num_workers_evaluation": 12,
    "env.num_gpus": 1,
    "optimization.max_epochs": 2000,
    "optimization.weight_decay": 1.0e-5,
    "optimization.lr_choice": None,
    "optimization.lr_schedule": "polynomial_decay",
    "optimization.warmup_steps": 0.0,
    "optimization.patience": 20,
    "optimization.top_k": 3,
}
```

We run the experinments on one NVIDIA Tesla T4 GPU with 15360MiB memory.
All trained models, exported results and loggings can be accessible from https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/tabular_example_result.zip.
Use:
```bash
wget https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/tabular_example_result.zip
unzip tabular_example_result.zip
```
to download the our results.


### 5. Ablations on Numerical Embedding Architectures

We present ablations on `AutoMM FT-Transformer` with variours embedding architectures [2].

We support the composition of the following architectures: {'linear', 'shared_linear', 'autodis', 'positional', 'relu', 'leaky_relu', 'layernorm'}.

We can reproduce the following results by tuning `--embedding_arch` in `example_tabular.py`.

Datasets | ca | ad | he | ja | hi | al | ep | ye | co | ya | mi | Results
----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----
metrics | rmse | acc | acc | acc | acc | acc | acc | rmse | acc | rmse | rmse
["linear"] | 0.482 | 0.859 | 0.379 | 0.721 | 0.726 | 0.949 | RuntimeError | 8.891 | 0.963 | 0.769 | 0.761 | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/tabular_example_result.zip)
["linear", "relu"] | 0.477 | 0.859 | 0.370 | 0.721 | 0.726 | 0.951 | RuntimeError | 8.953 | 0.967 | 0.772 | 0.757 | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/tabular_example_result2.zip)
["linear", "leaky_relu"] | 0.473 | 0.858 | 0.370 | 0.722 | 0.725 | 0.947 | RuntimeError | 8.915 | 0.965 | 0.771 | 0.776 | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/tabular_example_result3.zip)
["linear", "relu", "linear"] | 0.468 | 0.858 | 0.374 | 0.721 | 0.723 | 0.951 | RuntimeError | 8.941 | 0.965 | 0.769\* | 0.770 | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/tabular_example_result4.zip)
["positional", "linear"] | 0.467 | 0.864 | 0.347 | 0.694 | 0.709 | 0.951 | RuntimeError | 9.120 | 0.967 | 0.773\* | 0.761 | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/tabular_example_result5.zip)
["positional", "linear", "relu"] | 0.465 | 0.866 | 0.343 | 0.688 | 0.704 | 0.947 | RuntimeError | 9.131 | 0.967 | 0.774\* | 0.760 | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/tabular_example_result6.zip)
["positional"] | 0.480 | 0.861 | 0.334 | 0.684 | 0.696 | 0.951 | RuntimeError | 9.189 | 0.967 | 0.774 | 0.765 | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/tabular_example_result7.zip)

\* denotes adjusting `env.per_gpu_batch_size` from `128` to `64` to support runing a larger model on our device.

### 6. Hyper-parameter optimization for FT_Transformer
Set `--mode` in [`example_tabular.py`](./example_tabular.py) to `single_hpo` to run the HPO for FTTransformer. 
The search spaces for FT_transformer are as follws:
- "model.numerical_transformer.ffn_dropout": tune.uniform(0.0, 0.5), 
- "model.numerical_transformer.attention_dropout": tune.uniform(0.0, 0.5),
- "model.numerical_transformer.residual_dropout": tune.uniform(0.0, 0.2),
- "optimization.learning_rate": tune.uniform(0.00001, 0.001),

with the tuning kwargs as follws:
```python
hyperparameter_tune_kwargs = {
        "searcher": 'random',
        "scheduler": 'FIFO',
        "num_trials": 50,
}
```

The results are as follws:
Dataset (metric) | w/o HPO | w/ HPO | model_configs | results 
----  | ----  | ----  | ----  | ----  
ca (rmse) | 0.482 | 0.579 | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/hpo/ca_hpo_config.yaml) | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/hpo/ca_hpo_result.zip)
he (acc) | 0.379 | 0.381 | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/hpo/he_hpo_config.yaml) | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/hpo/he_hpo_result.zip)
ja (acc) | 0.721 | 0.728 | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/hpo/ja_hpo_config.yaml) | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/hpo/ja_hpo_result.zip)
hi (acc) | 0.726 | 0.727 | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/hpo/hi_hpo_config.yaml) | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/hpo/hi_hpo_result.zip)
al (acc) | 0.949 | 0.953 | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/hpo/al_hpo_config.yaml) | [link](https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/hpo/al_hpo_result.zip)

---

### Reference
[1] Revisiting Deep Learning Models for Tabular Data, 2021, https://arxiv.org/pdf/2106.11959.pdf

[2] On Embeddings for Numerical Features in Tabular Deep Learning, 2022, https://arxiv.org/abs/2203.05556
