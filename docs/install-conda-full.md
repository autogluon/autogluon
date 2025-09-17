```console
conda create -n ag python=3.10
conda activate ag
conda install -c conda-forge mamba
mamba install -c conda-forge autogluon
mamba install -c conda-forge "ray-tune >=2.10.0,<2.49" "ray-default >=2.10.0,<2.49"  # install ray for faster training
```
