

<div align="center">
<img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">

## Fast and Accurate ML in 3 Lines of Code

[![Latest Release](https://img.shields.io/github/v/release/autogluon/autogluon)](https://github.com/autogluon/autogluon/releases)
[![Conda Forge](https://img.shields.io/conda/vn/conda-forge/autogluon.svg)](https://anaconda.org/conda-forge/autogluon)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/autogluon/)
[![Downloads](https://pepy.tech/badge/autogluon/month)](https://pepy.tech/project/autogluon)
[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Discord](https://img.shields.io/discord/1043248669505368144?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.gg/wjUmjqAc2N)
[![Twitter](https://img.shields.io/twitter/follow/autogluon?style=social)](https://twitter.com/autogluon)
[![Continuous Integration](https://github.com/autogluon/autogluon/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/autogluon/autogluon/actions/workflows/continuous_integration.yml)
[![Platform Tests](https://github.com/autogluon/autogluon/actions/workflows/platform_tests-command.yml/badge.svg?event=schedule)](https://github.com/autogluon/autogluon/actions/workflows/platform_tests-command.yml)

[Installation](https://auto.gluon.ai/stable/install.html) | [Documentation](https://auto.gluon.ai/stable/index.html) | [Release Notes](https://auto.gluon.ai/stable/whats_new/index.html)

</div>

AutoGluon, developed by AWS AI, automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications.  With just a few lines of code, you can train and deploy high-accuracy machine learning and deep learning models on image, text, time series, and tabular data.


## 💾 Installation

AutoGluon is supported on Python 3.9 - 3.12 and is available on Linux, MacOS, and Windows.

You can install AutoGluon with:

```python
pip install autogluon
```

Visit our [Installation Guide](https://auto.gluon.ai/stable/install.html) for detailed instructions, including GPU support, Conda installs, and optional dependencies.

## :zap: Quickstart

Build accurate end-to-end ML models in just 3 lines of code!

```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label="class").fit("train.csv", presets="best")
predictions = predictor.predict("test.csv")
```

| AutoGluon Task      |                                                                                Quickstart                                                                                |                                                                                API                                                                                |
|:--------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| TabularPredictor    | [![Quick Start](https://img.shields.io/static/v1?label=&message=tutorial&color=grey)](https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html) |                 [![API](https://img.shields.io/badge/api-reference-blue.svg)](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html)                 |
| TimeSeriesPredictor | [![Quick Start](https://img.shields.io/static/v1?label=&message=tutorial&color=grey)](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html)            | [![API](https://img.shields.io/badge/api-reference-blue.svg)](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html) |
| MultiModalPredictor | [![Quick Start](https://img.shields.io/static/v1?label=&message=tutorial&color=grey)](https://auto.gluon.ai/stable/tutorials/multimodal/multimodal_prediction/multimodal-quick-start.html)            | [![API](https://img.shields.io/badge/api-reference-blue.svg)](https://auto.gluon.ai/stable/api/autogluon.multimodal.MultiModalPredictor.html) |

## :mag: Resources

### Hands-on Tutorials / Talks

Below is a curated list of recent tutorials and talks on AutoGluon. A comprehensive list is available [here](AWESOME.md#videos--tutorials).

| Title                                                                                                                    | Format   | Location                                                                         | Date       |
|--------------------------------------------------------------------------------------------------------------------------|----------|----------------------------------------------------------------------------------|------------|
| :tv: [AutoGluon: Towards No-Code Automated Machine Learning](https://www.youtube.com/watch?v=SwPq9qjaN2Q)                | Tutorial | [AutoML 2024](https://2024.automl.cc/)                                           | 2024/09/09 |
| :tv: [AutoGluon 1.0: Shattering the AutoML Ceiling with Zero Lines of Code](https://www.youtube.com/watch?v=5tvp_Ihgnuk) | Tutorial | [AutoML 2023](https://2023.automl.cc/)                                           | 2023/09/12 |
| :sound: [AutoGluon: The Story](https://automlpodcast.com/episode/autogluon-the-story)                                    | Podcast  | [The AutoML Podcast](https://automlpodcast.com/)                                 | 2023/09/05 |
| :tv: [AutoGluon: AutoML for Tabular, Multimodal, and Time Series Data](https://youtu.be/Lwu15m5mmbs?si=jSaFJDqkTU27C0fa) | Tutorial | PyData Berlin                                                                    | 2023/06/20 |
| :tv: [Solving Complex ML Problems in a few Lines of Code with AutoGluon](https://www.youtube.com/watch?v=J1UQUCPB88I)    | Tutorial | PyData Seattle                                                                   | 2023/06/20 |
| :tv: [The AutoML Revolution](https://www.youtube.com/watch?v=VAAITEds-28)                                                | Tutorial | [Fall AutoML School 2022](https://sites.google.com/view/automl-fall-school-2022) | 2022/10/18 |

### Scientific Publications
- [AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/pdf/2003.06505.pdf) (*Arxiv*, 2020) ([BibTeX](CITING.md#general-usage--autogluontabular))
- [Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation](https://proceedings.neurips.cc/paper/2020/hash/62d75fb2e3075506e8837d8f55021ab1-Abstract.html) (*NeurIPS*, 2020) ([BibTeX](CITING.md#tabular-distillation))
- [Benchmarking Multimodal AutoML for Tabular Data with Text Fields](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/9bf31c7ff062936a96d3c8bd1f8f2ff3-Paper-round2.pdf) (*NeurIPS*, 2021) ([BibTeX](CITING.md#autogluonmultimodal))
- [XTab: Cross-table Pretraining for Tabular Transformers](https://proceedings.mlr.press/v202/zhu23k/zhu23k.pdf) (*ICML*, 2023)
- [AutoGluon-TimeSeries: AutoML for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2308.05566) (*AutoML Conf*, 2023) ([BibTeX](CITING.md#autogluontimeseries))
- [TabRepo: A Large Scale Repository of Tabular Model Evaluations and its AutoML Applications](https://arxiv.org/pdf/2311.02971.pdf) (*AutoML Conf*, 2024)
- [AutoGluon-Multimodal (AutoMM): Supercharging Multimodal AutoML with Foundation Models](https://arxiv.org/pdf/2404.16233) (*AutoML Conf*, 2024) ([BibTeX](CITING.md#autogluonmultimodal))
- [Multi-layer Stack Ensembles for Time Series Forecasting](https://arxiv.org/abs/2511.15350) (*AutoML Conf*, 2025) ([BibTeX](CITING.md#autogluontimeseries))
- [Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821) (*Arxiv*, 2025) ([BibTeX](CITING.md#autogluontimeseries))

### Articles
- [AutoGluon-TimeSeries: Every Time Series Forecasting Model In One Library](https://towardsdatascience.com/autogluon-timeseries-every-time-series-forecasting-model-in-one-library-29a3bf6879db) (*Towards Data Science*, Jan 2024)
- [AutoGluon for tabular data: 3 lines of code to achieve top 1% in Kaggle competitions](https://aws.amazon.com/blogs/opensource/machine-learning-with-autogluon-an-open-source-automl-library/) (*AWS Open Source Blog*, Mar 2020)
- [AutoGluon overview & example applications](https://towardsdatascience.com/autogluon-deep-learning-automl-5cdb4e2388ec?source=friends_link&sk=e3d17d06880ac714e47f07f39178fdf2) (*Towards Data Science*, Dec 2019)

### Train/Deploy AutoGluon in the Cloud
- [AutoGluon Cloud](https://auto.gluon.ai/cloud/stable/index.html) (Recommended)
- [AutoGluon on SageMaker AutoPilot](https://auto.gluon.ai/stable/tutorials/cloud_fit_deploy/autopilot-autogluon.html)
- [AutoGluon on Amazon SageMaker](https://auto.gluon.ai/stable/tutorials/cloud_fit_deploy/cloud-aws-sagemaker-train-deploy.html)
- [AutoGluon Deep Learning Containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#autogluon-training-containers) (Security certified & maintained by the AutoGluon developers)
- [AutoGluon Official Docker Container](https://hub.docker.com/r/autogluon/autogluon)
- [AutoGluon-Tabular on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-n4zf5pmjt7ism) (Not maintained by us)

## :pencil: Citing AutoGluon

If you use AutoGluon in a scientific publication, please refer to our [citation guide](CITING.md).

## :wave: How to get involved

We are actively accepting code contributions to the AutoGluon project. If you are interested in contributing to AutoGluon, please read the [Contributing Guide](https://github.com/autogluon/autogluon/blob/master/CONTRIBUTING.md) to get started.

## :classical_building: License

This library is licensed under the Apache 2.0 License.
