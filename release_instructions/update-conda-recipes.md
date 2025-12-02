# How to update the conda-forge recipes for AutoGluon

## Conda-forge packages

- [autogluon.common-feedstock](https://anaconda.org/conda-forge/autogluon.common)
- [autogluon.features-feedstock](https://anaconda.org/conda-forge/autogluon.features)
- [autogluon.core-feedstock](https://anaconda.org/conda-forge/autogluon.core)
- [autogluon.tabular-feedstock](https://anaconda.org/conda-forge/autogluon.tabular)
- [autogluon.multimodal-feedstock](https://anaconda.org/conda-forge/autogluon.multimodal)
- [autogluon.timeseries-feedstock](https://anaconda.org/conda-forge/autogluon.timeseries)
- [autogluon-feedstock](https://anaconda.org/conda-forge/autogluon)

## Conda-forge recipes

- [autogluon.common-feedstock](https://github.com/conda-forge/autogluon.common-feedstock)
- [autogluon.features-feedstock](https://github.com/conda-forge/autogluon.features-feedstock/)
- [autogluon.core-feedstock](https://github.com/conda-forge/autogluon.core-feedstock)
- [autogluon.tabular-feedstock](https://github.com/conda-forge/autogluon.tabular-feedstock)
- [autogluon.multimodal-feedstock](https://github.com/conda-forge/autogluon.multimodal-feedstock)
- [autogluon.timeseries-feedstock](https://github.com/conda-forge/autogluon.timeseries-feedstock)
- [autogluon-feedstock](https://github.com/conda-forge/autogluon-feedstock)

## Sample PRs

- [autogluon.common-feedstock](https://github.com/conda-forge/autogluon.common-feedstock/pull/6/files)
- [autogluon.features-feedstock](https://github.com/conda-forge/autogluon.features-feedstock/pull/5/files)
- [autogluon.core-feedstock](https://github.com/conda-forge/autogluon.core-feedstock/pull/8/files)
- [autogluon.tabular-feedstock](https://github.com/conda-forge/autogluon.tabular-feedstock/pull/15/files)
- [autogluon.multimodal-feedstock](https://github.com/conda-forge/autogluon.multimodal-feedstock/pull/16/files)
- [autogluon.timeseries-feedstock](https://github.com/conda-forge/autogluon.timeseries-feedstock/pull/7/files)
- [autogluon-feedstock](https://github.com/conda-forge/autogluon-feedstock/pull/6/files)

## Steps to update the conda-forge recipes

1. Go to the [AutoGluon](https://github.com/autogluon/autogluon/releases) release page on GitHub. Under the `Assets` section, right click on `Source code (tar.gz)` to copy the link address. It should be something like this:

   ```text
   https://github.com/autogluon/autogluon/archive/refs/tags/v0.7.0.tar.gz
   ```

2. Click on the link above to download the source code. The file should be named something like `autogluon-0.7.0.tar.gz`.
3. Use `openssl` to generate the sha256 hash of the downloaded file, e.g.,

   ```bash
   openssl sha256 autogluon-0.7.0.tar.gz
   ```

   The output should be something like this:

   ```text
   455831de3c9de8fbe11b100054b8f150661d0651212fcfa4ec2e42417fdac355
   ```

4. Fork the [autogluon.common-feedstock](https://github.com/conda-forge/autogluon.common-feedstock) repo to your GitHub account.
5. Clone the forked repo to your local machine.
6. Create a new branch, e.g., `v0.7.0`
7. Open the `recipe/meta.yaml` file in your favorite text editor. Update the `version`, `sha256` and `number` fields. The `version` field should be the version number of the new release. For example, if the new release is `v0.7.0`, then the `version` field should be `0.7.0`. The `sha256` field should be the hash generated in step 3. The `number` field should be reset to `0` for a new release. If the `version` number stays the same, then the `number` field should be incremented by 1. This is usually the case when you are updating the dependencies of the package but not updating the package version.

![](https://i.imgur.com/3hvO7z9.png)

8. Update the package dependencies and version bounds for each recipe based on the release. For `autogluon.common`, the dependency list can be found at [`common/setup.py`](https://github.com/autogluon/autogluon/blob/master/common/setup.py#L19), but the version bounds can be found at [`core/_setup_utils.py`](https://github.com/autogluon/autogluon/blob/master/core/src/autogluon/core/_setup_utils.py#L20)

![](https://i.imgur.com/MT8xe3Y.png)

9. Commit the changes and push to your forked repo. Then create a pull request to the `autogluon.common-feedstock` repo.
10. Comment on the pull request with the following text to trigger the CI build:

    ```text
    @conda-forge-admin, please rerender
    ```

11. Once the CI build is successful, merge the pull request.
12. Repeat steps above for the other six packages.

## Steps to build the conda-forge packages locally

Optionally, you can build the conda-forge packages locally to test if the recipes are correct. This is especially useful when you are updating the dependencies of the package. The steps are as follows:

1. Install docker on your machine. See [here](https://docs.docker.com/get-docker/) for instructions.
2. Install Anaconda or Miniconda on your machine. See [here](https://docs.conda.io/en/latest/miniconda.html) for instructions.
3. Install `mamba` in the base environment of your conda installation. This is a faster version of `conda` that is recommended for conda-forge builds.

   ```bash
   conda install -n base mamba -c conda-forge
   ```

4. Create a new conda environment named `ag` with Python 3.10 or higher that's supported by AutoGluon.

   ```bash
   mamba create -n ag python=3.10
   ```

5. Navigate to the root directory of the cloned repo of the package you want to build. For example, if you want to build `autogluon.multimodal`, then you should be in the root directory of the `autogluon.multimodal-feedstock` repo.

   ```bash
    chmod 777 -R autogluon.multimodal-feedstock
    cd autogluon.multimodal-feedstock
    python build-locally.py
   ```

6. Choose an option that you want to build that matches your computer configuration. For example, if you want to build the `linux_64_cuda` package with Python 3.10, then choose option `3`.

![](https://i.imgur.com/xKtQyS4.png)

7. If the build is successful, you should find the built packages in the `build_artifacts` directory under the root directory of the cloned repo.
8. Install the built packages in the `ag` environment you created in step 4.

   ```bash
    mamba install -n ag -c "file://${PWD}/build_artifacts" -c conda-forge  autogluon.multimodal
   ```

   Make sure to check the `pytorch` version in the list of packages to install. If you see `cuda` in the version number, that means the package is built with CUDA support. Otherwise, it's built without CUDA support.

![](https://i.imgur.com/mpGM1pV.png)

## How to add maintainers to the conda-forge recipes

1. Please ask the existing maintainers if you want to be added as a maintainer. Only the existing maintainers can add new maintainers.
2. The existing maintainer needs to go to the feedstock repo, e.g., [autogluon.common-feedstock](https://github.com/conda-forge/autogluon-feedstock/issues/new/choose).
3. Open a new issue and choose the `Bot commands` template. Click on `Get started` to open the issue.
4. Enter the following text in the title of the issue. Be sure to replace `@username` with the GitHub username of the maintainer you want to add.

```bash
@conda-forge-admin, please add user @username
```

5. Click on `Submit new issue` to submit the issue.
6. Once the issue is submitted and the CI build is successful, merge the pull request.
