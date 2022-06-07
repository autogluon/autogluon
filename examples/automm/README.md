Examples showing how to use AutoMMPredictor.


## 1. Tabular Data

### 1.1 Example
[`example_tabular.py`](./example_tabular.py) : This example provide a use case for the pure *tabular* data, including numerical and categorical, with FT_Transformer [1].

To run the example: 

```python example_tabular.py --dataset_name ad --dataset_dir ./dataset --exp_dir ./result```
   - `dataset_name` determines which dataset to run the experinments, refers to [Dataset Section](###1.2-Datasets).
   - `dataset_dir` is the path to the dataset(s). If the datasets do not present in this path, it will be automatically downloaded.
   - `exp_dir` is the output path to store the weights and loggings. 
   - `seed` determines the random seed. Default is 0.


### 1.2 Datasets
We borrow 11 tabular datasets provided by [1], and use identically the same abbreviation as Table 1 in [1] to name each datasets. 
The original datasets provided by https://github.com/Yura52/tabular-dl-revisiting-models are all in  `Numpy.darray` format (can be downloaded from https://www.dropbox.com/s/o53umyg6mn3zhxy/data.tar.gz?dl=1). 
Data in `Numpy.darray` was first pre-processedin into `.csv` format, which can be loaded by `pandas.Dataframe` as the input to AutoMMPredictor. 
All Data will be automatically downloaded from s3 (thus online connection is necessary) if it does not exisit with the given dataset path. 


### 1.3 FT_Transformer
We categorize the original FT_Transformer to two models in `AutoMMPRedictor`, namely `numerical_transformer` and `categorical transformer`, which depends on the modaility of input tabular data (i.e., numerical v.s. categorical). The two models share most of the common features:
   - `out_features` is the output feature size.
   - `d_token` is the dimension of the embedding tokens.
   - `num_trans_blocks` is the number of transformer blocks.
   - `num_attn_heads` is the number of the attention heads in multi-headed self attention layer in each transformer block.
   - `ffn_dropout` is the dropout rate in feadforward layer.
   - `ffn_activation` determines the activation fuction in feadforward layer. We support `relu`, `gelu`, `reglu` and `leaky_relu`.
   - `attention_dropout` is the dropout rate in attention layer.
`numerical_transformer` supports an additional feature:
   - `embedding_arch` is a list containing the names of embedding layers as described in [2]. Currently we support the following embedding layers: {'linear', 'shared_linear', 'autodis', 'positional', 'relu', 'layernorm'}. Whatever the embedding layers are selected, the shape of the output embedding is `batch_size * number_of_numerical_features * d_token`.
  
These features can be tuned using `hyperparameters` in `AutoMMPredictor. For example: 
```
hyperparameters = {
   'model.names': ["categorical_transformer","numerical_transformer","fusion_transformer"],
   'model.categorical_transformer.num_trans_blocks': 1,
   'ffn_dropout': 0.0,
}
```


### 1.4 Results
Datasets | ca | ad | he | ja | hi | al | ep | ye | co | ya | ml 
----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  
metrics | rmse | acc | acc | acc | acc | acc | acc | rmse | acc | rmse | rmse
problem_type | regression | binary | multiclass | multiclass | binary | multiclass | binary | regression | multiclass | regression | regression
#objects | 20640 | 48842 | 65196 | 83733 | 98050 | 108000 | 500000 | 515345 | 581012 | 709877 | 1200192
#num. | 8 | 6 | 27 | 54 | 28 | 128 | 2000 | 90 | 54 | 699 | 136
#cat. | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
#classes | - | 2 | 100 | 4 | 2 | 1000 | 2 | - | 7 | - | -
Best in [1] | 0.459 | 0.859 | 0.396 | 0.732 | 0.729 | 0.963 | 0.8982 | 8.794 | 0.970 | 0.753 | 0.745
FT-Transformer in [1] | 0.459 | 0.859 | 0.391 | 0.732 | 0.729 | 0.960 | 0.8982 | 8.855 | 0.970 | 0.756 | 0.746
AutoMM FT-Transformer + Const Lr=1e-4 | 0.404 | 0.863 | 0.388 | 0.727 | 0.729 | 0.954 | OverflowError | 0.781 | 0.964 | AttributeError | 0.904
AutoMM FT-Transformer + Cosine Lr=1e-4| 0.401 | 0.864 | 0.388 | 0.728 | 0.729 | 0.952 | OverflowError | 0.780 | 0.964 | 0.765 | 0.924

The trained weights and configs are accessible at https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/results.zip or s3://autogluon/results/tabular/results.zip. Use
```
wget https://autogluon.s3.us-west-2.amazonaws.com/results/tabular/results.zip 
unzip results.zip
```
to download the results.


### Reference
[1]: Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, 
    "Revisiting Deep Learning Models for Tabular Data", 2021. 
    https://arxiv.org/pdf/2106.11959.pdf

[2] On Embeddings for Numerical Features in Tabular Deep Learning, https://arxiv.org/abs/2203.05556
