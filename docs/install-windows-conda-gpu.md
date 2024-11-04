```console
conda create -n ag python=3.11
conda activate ag
conda install -c conda-forge mamba
mamba install -c conda-forge -c pytorch -c nvidia autogluon "pytorch=*=*cuda*"
mamba install -c conda-forge "ray-tune >=2.10.0,<2.32" "ray-default >=2.10.0,<2.32"  # install ray for faster training
```
