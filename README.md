

<div align="left">
  <img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</div>

## AutoML for Text, Image, and Tabular Data

[![Build Status](https://ci.gluon.ai/view/all/job/autogluon/job/master/badge/icon)](https://ci.gluon.ai/view/all/job/autogluon/job/master/)
[![Pypi Version](https://img.shields.io/pypi/v/autogluon.svg)](https://pypi.org/project/autogluon/#history)
![Upload Python Package](https://github.com/awslabs/autogluon/workflows/Upload%20Python%20Package/badge.svg)

AutoGluon automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications.  With just a few lines of code, you can train and deploy high-accuracy machine learning and deep learning models on text, image, and tabular data.

## Example

```python
# First install package from terminal:
# python3 -m pip install --upgrade pip
# python3 -m pip install --upgrade setuptools
# python3 -m pip install --upgrade "mxnet<2.0.0"
# python3 -m pip install --pre autogluon

from autogluon.tabular import TabularDataset, TabularPredictor
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
predictor = TabularPredictor(label='class').fit(train_data, time_limit=60)  # Fit models for 60s
leaderboard = predictor.leaderboard(test_data)
```
## News

**Announcement for previous users:** The AutoGluon codebase has been modularized into [namespace packages](https://packaging.python.org/guides/packaging-namespace-packages/), which means you now only need those dependencies relevant to your prediction task of interest! For example, you can now work with tabular data without having to [install](https://auto.gluon.ai/dev/install.html) dependencies required for AutoGluon's computer vision tasks (and vice versa). Unfortunately this improvement required a minor API change (eg. instead of `from autogluon import TabularPrediction`, you should now do: `from autogluon.tabular import TabularPredictor`), for all versions newer than v0.0.15. Documentation/tutorials under the old API may still be viewed [for version 0.0.15](https://auto.gluon.ai/0.0.15/index.html) which is the last released version under the old API.


## Resources

See the [AutoGluon Website](https://auto.gluon.ai/stable/index.html) for [documentation](https://auto.gluon.ai/stable/api/index.html) and instructions on:
- [Installing AutoGluon](https://auto.gluon.ai/stable/index.html#installation)
- [Learning with tabular data](https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-quickstart.html)
  - [Tips to maximize accuracy](https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-quickstart.html#maximizing-predictive-performance) (if **benchmarking**, make sure to run `fit()` with argument `presets='best_quality'`).  

- [Learning with text data](https://auto.gluon.ai/stable/tutorials/text_prediction/beginner.html)
- [Learning with image data](https://auto.gluon.ai/stable/tutorials/image_prediction/beginner.html)
- More advanced topics such as [Neural Architecture Search](https://auto.gluon.ai/stable/tutorials/nas/index.html)

### Scientific Publications
- [AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/pdf/2003.06505.pdf) (*Arxiv*, 2020)

### Articles
- [AutoGluon for tabular data: 3 lines of code to achieve top 1% in Kaggle competitions](https://aws.amazon.com/blogs/opensource/machine-learning-with-autogluon-an-open-source-automl-library/) (*AWS Open Source Blog*, Mar 2020)
- [Accurate image classification in 3 lines of code with AutoGluon](https://medium.com/@zhanghang0704/image-classification-on-kaggle-using-autogluon-fc896e74d7e8) (*Medium*, Feb 2020)
- [AutoGluon overview & example applications](https://towardsdatascience.com/autogluon-deep-learning-automl-5cdb4e2388ec?source=friends_link&sk=e3d17d06880ac714e47f07f39178fdf2) (*Towards Data Science*, Dec 2019)

### Hands-on Tutorials
- [From HPO to NAS: Automated Deep Learning (CVPR 2020)](https://hangzhang.org/CVPR2020/)
- [Practical Automated Machine Learning with Tabular, Text, and Image Data (KDD 2020)](https://jwmueller.github.io/KDD20-tutorial/)

### Train/Deploy AutoGluon in the Cloud
- [AutoGluon-Tabular on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-n4zf5pmjt7ism)
- [Running AutoGluon-Tabular on Amazon SageMaker](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/autogluon-tabular/AutoGluon_Tabular_SageMaker.ipynb)
- [Running AutoGluon Image Classification on Amazon SageMaker](https://github.com/zhanghang1989/AutoGluon-Docker)

## Citing AutoGluon

If you use AutoGluon in a scientific publication, please cite the following paper:

Erickson, Nick, et al. ["AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data."](https://arxiv.org/abs/2003.06505) arXiv preprint arXiv:2003.06505 (2020).

BibTeX entry:

```bibtex
@article{agtabular,
  title={AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data},
  author={Erickson, Nick and Mueller, Jonas and Shirkov, Alexander and Zhang, Hang and Larroy, Pedro and Li, Mu and Smola, Alexander},
  journal={arXiv preprint arXiv:2003.06505},
  year={2020}
}
```

## AutoGluon for Hyperparameter and Neural Architecture Search (HNAS)

AutoGluon also provides state-of-the-art tools for neural hyperparameter and architecture search, such as for example ASHA, Hyperband, Bayesian Optimization and BOHB. To get started, checkout the following resources

- [General introduction into HNAS](https://www.youtube.com/watch?v=pB1LmZWK_N8&feature=youtu.be)
- [Introduction into HNAS with AutoGluon](https://www.youtube.com/watch?v=GJVwUyVWZas)
- [Example notebook](https://github.com/zhanghang1989/HPO2NAS-Tutorial-CVPR-ECCV2020/blob/master/mlp.ipynb)
- [Example scripts for efficient multi-fidelity HNAS of PyTorch neural network models](https://github.com/awslabs/autogluon/tree/master/examples/hnas/)

Also have a look at our paper ["Model-based Asynchronous Hyperparameter and Neural Architecture Search"](https://arxiv.org/abs/2003.10865) arXiv preprint arXiv:2003.10865 (2020).

```bibtex
@article{abohb,
  title={Model-based Asynchronous Hyperparameter and Neural Architecture Search},
  author={Klein, Aaron and Tiao, Louis and Lienart, Thibaut and Archambeau, Cedric and Seeger, Matthias},
  journal={arXiv preprint arXiv:2003.10865},
  year={2020}
}
```

## License

This library is licensed under the Apache 2.0 License.

## Contributing to AutoGluon

We are actively accepting code contributions to the AutoGluon project. If you are interested in contributing to AutoGluon, please read the [Contributing Guide](https://github.com/awslabs/autogluon/blob/master/CONTRIBUTING.md) to get started.
