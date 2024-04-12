```console
conda create -n ag python=3.10
conda activate ag
conda install -c conda-forge mamba
mamba install -c conda-forge autogluon "pytorch=*=cuda*"
mamba install -c conda-forge "ray-tune >=2.6.3,<2.7" "ray-default >=2.6.3,<2.7"  # install ray for faster training
```
