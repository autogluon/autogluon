:::{note}
The `autogluon.multimodal` conda-forge package does not yet support Windows. See the following instructions to install `autogluon.tabular` and `autogluon.timeseries` on Windows:
:::

```console
conda create -n ag python=3.9
conda activate ag
conda install -c conda-forge mamba
mamba install -c conda-forge autogluon.tabular autogluon.timeseries
```
