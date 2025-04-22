# Installing AutoGluon

:::{note}

* AutoGluon requires Python version 3.9, 3.10, 3.11, 3.12, or 3.13 and is available on Linux, MacOS, and Windows.

* The AutoGluon library comes pre-installed in all releases of [Amazon SageMaker Distribution](https://github.com/aws/sagemaker-distribution). For more information, refer to the dropdown [AutoGluon in Amazon SageMaker Studio](#dropdown-sagemaker) in this page.

We recommend most users to install via pip. The pip install of AutoGluon is the version we actively benchmark and test on.
The Conda install may have subtle differences in installed dependencies that could impact performance and stability, and we recommend trying pip if you run into issues with Conda.

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

  :::::{tab} UV

    ::::{tab} CPU
    ```{include} install-cpu-uv.md
    ```
    ::::

    ::::{tab} GPU
    ```{include} install-gpu-uv.md
    ```
    ::::

  :::::

  :::::{tab} Conda

    ::::{tab} CPU
    ```{include} install-conda-full.md
    ```
    ::::

    ::::{tab} GPU
    ```{include} install-linux-conda-gpu.md
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

  :::::{tab} UV

    ::::{tab} CPU
    ```{include} install-mac-libomp.md
    ```

    ```{include} install-cpu-uv.md
    ```
    ::::

    ::::{tab} GPU
    ```{include} install-mac-nogpu.md
    ```
    ::::

  :::::

  :::::{tab} Conda

    ::::{tab} CPU
    ```{include} install-mac-conda.md
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

    ```{include} install-mac-cpu-source.md
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

  :::::{tab} UV

    ::::{tab} CPU
    ```{include} install-windows-cpu.md
    ```

    ```{include} install-cpu-uv.md
    ```
    ::::

    ::::{tab} GPU
    ```{include} install-windows-gpu.md
    ```

    ```{include} install-gpu-uv.md
    ```
    ::::

  :::::

  :::::{tab} Conda

    ::::{tab} CPU
    ```{include} install-conda-full.md
    ```
    ::::

    ::::{tab} GPU
    ```{include} install-windows-conda-gpu.md
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

<div id="dropdown-sagemaker"></div>

:::{dropdown} AutoGluon in Amazon SageMaker Studio

[Amazon SageMaker Distribution](https://github.com/aws/sagemaker-distribution) is the docker environment for data science used as the default image of [JupyterLab](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl.html) notebook instances and [Code Editor](https://docs.aws.amazon.com/sagemaker/latest/dg/code-editor.html) in [Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated.html).  The AutoGluon library comes pre-installed in all releases of Amazon SageMaker Distribution. SageMaker Studio users can access AutoGluon's automation capabilities without needing to install anything additional.


To find the AutoGluon and PyTorch versions available in a SageMaker Distribution image, refer to the [RELEASE.md](https://github.com/aws/sagemaker-distribution/blob/main/build_artifacts/v1/v1.4/v1.4.2/RELEASE.md) file for your image version in the SageMaker Distribution GitHub repository.

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
./autogluon/full_install.sh
```

Note that the above example is only valid while the branch still exists. A user could delete the branch after the PR is merged, so this advice is primarily focused on unmerged PRs.

:::


:::{dropdown} Install nightly builds

AutoGluon offers nightly builds that can be installed using the `--pre` argument. Nightly builds have the latest features but have not been as rigorously tested as stable releases.

```bash
pip install -U uv
python -m uv pip install --pre autogluon
```
:::


:::{dropdown} M1 and M2 Apple Silicon

Apple Silicon is now supported via the `conda` installation instructions outlined above. `conda-forge` will install the GPU version if a user's machine supports it.

:::


:::{dropdown} Kaggle

AutoGluon is actively used by the Kaggle community. You can find hundreds of Kaggle notebooks using AutoGluon [here](https://www.kaggle.com/search?q=autogluon+in%3Anotebooks+sortBy%3Adate).

For Kaggle competitions that allow internet access in notebooks, you can install AutoGluon via the following line at the start of the notebook:

```
!pip install -U autogluon > /dev/null
```

For competitions without internet access, you can obtain AutoGluon by using [one of the Kaggle community's packaged AutoGluon artifacts](https://www.kaggle.com/search?q=autogluon+in%3Adatasets+sortBy%3Adate) in the form of a Kaggle dataset.

If you encounter issues after installing AutoGluon, try restarting the notebook runtime to ensure a clean memory state.

:::


:::{admonition} Trouble Shooting

If you encounter installation issues not covered here, please create a [GitHub issue](https://github.com/autogluon/autogluon/issues).

:::
