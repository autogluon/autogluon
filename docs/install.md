# Installing AutoGluon

:::{note} 

AutoGluon requires Python version 3.8, 3.9, or 3.10 and is available on Linux, MacOS, and Windows.

:::

::::::{tab} Linux

  :::::{tab} Pip
  
    ::::{tab} CPU
    ```{include} install-cpu-pip.md
    ```
    ::::
    
    ::::{tab} GPU
    ```{include} install-gpu-pip.md
    ```
    ::::
  
  :::::
  
  :::::{tab} Conda
  
    ::::{tab} CPU
    ```{include} install-conda-full.md
    ```
    ::::
    
    ::::{tab} GPU
    ```{include} install-conda-full.md
    ```
    ::::
  
  :::::
  
  :::::{tab} Source
  
    ::::{tab} CPU
    ```{include} install-cpu-source.md
    ```
    ::::
    
    ::::{tab} GPU
    ```{include} install-gpu-source.md
    ```
    ::::
  
  :::::
  
::::::

::::::{tab} Mac

  :::::{tab} Pip
  
    ::::{tab} CPU
    ```{include} install-mac-libomp.md
    ```

    ```{include} install-mac-cpu.md 
    ``` 
    ::::
  
    ::::{tab} GPU
    ```{include} install-mac-nogpu.md
    ::::
  
  :::::
  
  :::::{tab} Conda
  
    ::::{tab} CPU
    ```{include} install-conda-full.md
    ```
    ::::
  
    ::::{tab} GPU
    ```{include} install-mac-nogpu.md
    ```
    ::::
  
  :::::
  
  :::::{tab} Source
  
    ::::{tab} CPU
    ```{include} install-mac-libomp.md
    ```

    ```{include} install-cpu-source.md
    ```
    ::::
    
    ::::{tab} GPU
    ```{include} install-mac-nogpu.md
    ```
    ::::
  
  :::::

::::::

::::::{tab} Windows

  :::::{tab} Pip
  
    ::::{tab} CPU
    ```{include} install-windows-cpu.md
    ```
    
	  ```{include} install-cpu-pip.md 
    ```
    ::::
    
    ::::{tab} GPU
    ```{include} install-windows-gpu.md
    ```

	  ```{include} install-gpu-pip.md 
    ```
    ::::
  
  :::::
  
  :::::{tab} Conda
  
    ::::{tab} CPU
    ```{include} install-windows-conda.md
    ```
    ::::
    
    ::::{tab} GPU
    ```{include} install-windows-conda.md
    ```
    ::::
  
  :::::
  
  :::::{tab} Source
  
    ::::{tab} CPU
    ```{include} install-windows-cpu.md
    ```

    ```{include} install-cpu-source.md
    ```
    ::::
    
    ::::{tab} GPU
    ```{include} install-windows-gpu.md
    ```

    ```{include} install-gpu-source.md
    ```
    ::::
  
  :::::

::::::

:::{dropdown} Install specific AutoGluon modules and dependencies

```{include} install-modules.md
```

:::


:::{dropdown} Install from source for a specific pull-request

To build AutoGluon from source for the purposes of testing a pull-request, you can clone and install the exact branch by following these instructions.
This process is useful if you are a code reviewer or want to test if a PR fixes a bug you have reported.

In this example, we are using [this PR](https://github.com/autogluon/autogluon/pull/2944).
It is from the user `innixma` and the PR branch is called `accel_preprocess_bool`. 
This information is provided in the PR page directly under the title of the PR (where it says `into autogluon:master from Innixma:accel_preprocess_bool`).

```bash
# Edit these two variables to change which PR / branch is being installed
GITHUB_USER=innixma
BRANCH=accel_preprocess_bool

pip install -U pip
git clone --depth 1 --single-branch --branch ${BRANCH} --recurse-submodules https://github.com/${GITHUB_USER}/autogluon.git
cd autogluon && ./full_install.sh
```

Note that the above example is only valid while the branch still exists. A user could delete the branch after the PR is merged, so this advice is primarily focused on unmerged PRs.

:::


:::{dropdown} Install nightly builds

AutoGluon offers nightly builds that can be installed using the `--pre` argument. Nightly builds have the latest features but have not been as rigorously tested as stable releases.

```bash
pip install --pre autogluon
```
:::


:::{dropdown} M1 and M2 Apple Silicon

Apple Silicon is now supported via the `conda` installation instructions outlined above. `conda-forge` will install the GPU version if a user's machine supports it.

:::


:::{dropdown} Kaggle

AutoGluon dropped Python 3.7 support in v0.7. However, the Kaggle container's default Python version is still 3.7, which will lead to AutoGluon installation issues.
To upgrade the Python version to 3.8 or higher, a workaround is described here: [Alternative Python Version (Hack)](https://www.kaggle.com/code/amareltaylor/how-to-install-alternative-python-version-hack).

```bash
conda create -n newPython -c cctbx202208 -y
source /opt/conda/bin/activate newPython && conda install -c cctbx202208 python -y
/opt/conda/envs/newPython/bin/python3 -m pip install autogluon
```

Once AutoGluon is installed, restart the notebook runtime and import modules before running AutoGluon code.

:::


:::{admonition} Trouble Shooting

If you encounter installation issues not covered here, please create a [GitHub issue](https://github.com/autogluon/autogluon/issues).

:::

