

<div align="left">
  <img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</div>

## AutoML for Image, Text, and Tabular Data

[![Latest Release](https://img.shields.io/github/v/release/awslabs/autogluon)](https://github.com/awslabs/autogluon/releases)
[![Build Status](https://ci.gluon.ai/view/all/job/autogluon/job/master/badge/icon)](https://ci.gluon.ai/view/all/job/autogluon/job/master/)
[![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://pypi.org/project/autogluon/)
[![GitHub license](docs/static/apache2.svg)](./LICENSE)
[![Downloads](https://pepy.tech/badge/autogluon/month)](https://pepy.tech/project/autogluon)
[![Twitter](https://img.shields.io/twitter/follow/autogluon?style=social)](https://twitter.com/autogluon)

[Install Instructions](https://auto.gluon.ai/stable/install.html) | Documentation ([Stable](https://auto.gluon.ai/stable/index.html) | [Latest](https://auto.gluon.ai/dev/index.html))

AutoGluon automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications.  With just a few lines of code, you can train and deploy high-accuracy machine learning and deep learning models on image, text, and tabular data.

## Example

```python
# First install package from terminal:
# pip install -U pip
# pip install -U setuptools wheel
# pip install autogluon  # autogluon==0.4.0

from autogluon.tabular import TabularDataset, TabularPredictor
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
predictor = TabularPredictor(label='class').fit(train_data, time_limit=120)  # Fit models for 120s
leaderboard = predictor.leaderboard(test_data)
```

| AutoGluon Task | Quickstart | API |
| :--- | :---: | :---: |
| TabularPredictor | [![Quick Start](https://img.shields.io/static/v1?label=&message=tutorial&color=grey)](https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-quickstart.html) | [![API](https://img.shields.io/badge/api-reference-blue.svg)](https://auto.gluon.ai/stable/api/autogluon.predictor.html#module-0) |
| TextPredictor | [![Quick Start](https://img.shields.io/static/v1?label=&message=tutorial&color=grey)](https://auto.gluon.ai/stable/tutorials/text_prediction/beginner.html) | [![API](https://img.shields.io/badge/api-reference-blue.svg)](https://auto.gluon.ai/stable/api/autogluon.predictor.html#module-3) |
| ImagePredictor | [![Quick Start](https://img.shields.io/static/v1?label=&message=tutorial&color=grey)](https://auto.gluon.ai/stable/tutorials/image_prediction/beginner.html) | [![API](https://img.shields.io/badge/api-reference-blue.svg)](https://auto.gluon.ai/stable/api/autogluon.predictor.html#module-1) |
| ObjectDetector | [![Quick Start](https://img.shields.io/static/v1?label=&message=tutorial&color=grey)](https://auto.gluon.ai/stable/tutorials/object_detection/beginner.html) | [![API](https://img.shields.io/badge/api-reference-blue.svg)](https://auto.gluon.ai/stable/api/autogluon.predictor.html#module-2) |

## Resources

See the [AutoGluon Website](https://auto.gluon.ai/stable/index.html) for [documentation](https://auto.gluon.ai/stable/api/index.html) and instructions on:
- [Installing AutoGluon](https://auto.gluon.ai/stable/index.html#installation)
- [Learning with tabular data](https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-quickstart.html)
  - [Tips to maximize accuracy](https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-quickstart.html#maximizing-predictive-performance) (if **benchmarking**, make sure to run `fit()` with argument `presets='best_quality'`).  

- [Learning with text data](https://auto.gluon.ai/stable/tutorials/text_prediction/beginner.html)
- [Learning with image data](https://auto.gluon.ai/stable/tutorials/image_prediction/beginner.html)

Refer to the [AutoGluon Roadmap](https://github.com/awslabs/autogluon/blob/master/ROADMAP.md) for details on upcoming features and releases.

### Scientific Publications
- [AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/pdf/2003.06505.pdf) (*Arxiv*, 2020)
- [Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation](https://proceedings.neurips.cc/paper/2020/hash/62d75fb2e3075506e8837d8f55021ab1-Abstract.html) (*NeurIPS*, 2020)
- [Multimodal AutoML on Structured Tables with Text Fields](https://openreview.net/pdf?id=OHAIVOOl7Vl) (*ICML AutoML Workshop*, 2021)

### Articles
- [AutoGluon for tabular data: 3 lines of code to achieve top 1% in Kaggle competitions](https://aws.amazon.com/blogs/opensource/machine-learning-with-autogluon-an-open-source-automl-library/) (*AWS Open Source Blog*, Mar 2020)
- [Accurate image classification in 3 lines of code with AutoGluon](https://medium.com/@zhanghang0704/image-classification-on-kaggle-using-autogluon-fc896e74d7e8) (*Medium*, Feb 2020)
- [AutoGluon overview & example applications](https://towardsdatascience.com/autogluon-deep-learning-automl-5cdb4e2388ec?source=friends_link&sk=e3d17d06880ac714e47f07f39178fdf2) (*Towards Data Science*, Dec 2019)

### Hands-on Tutorials
- [Practical Automated Machine Learning with Tabular, Text, and Image Data (KDD 2020)](https://jwmueller.github.io/KDD20-tutorial/)

### Train/Deploy AutoGluon in the Cloud
- [AutoGluon-Tabular on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-n4zf5pmjt7ism)
- [AutoGluon-Tabular on Amazon SageMaker](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/autogluon-tabular-containers)
- [AutoGluon Deep Learning Containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#autogluon-training-containers)

## Contributing to AutoGluon

We are actively accepting code contributions to the AutoGluon project. If you are interested in contributing to AutoGluon, please read the [Contributing Guide](https://github.com/awslabs/autogluon/blob/master/CONTRIBUTING.md) to get started.

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

If you are using AutoGluon Tabular's model distillation functionality, please cite the following paper:

Fakoor, Rasool, et al. ["Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation."](https://proceedings.neurips.cc/paper/2020/hash/62d75fb2e3075506e8837d8f55021ab1-Abstract.html) Advances in Neural Information Processing Systems 33 (2020).

BibTeX entry:

```bibtex
@article{agtabulardistill,
  title={Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation},
  author={Fakoor, Rasool and Mueller, Jonas W and Erickson, Nick and Chaudhari, Pratik and Smola, Alexander J},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

If you use AutoGluon's multimodal text+tabular functionality in a scientific publication, please cite the following paper:

Shi, Xingjian, et al. ["Multimodal AutoML on Structured Tables with Text Fields."](https://openreview.net/forum?id=OHAIVOOl7Vl) 8th ICML Workshop on Automated Machine Learning (AutoML). 2021.

BibTeX entry:

```bibtex
@inproceedings{agmultimodaltext,
  title={Multimodal AutoML on Structured Tables with Text Fields},
  author={Shi, Xingjian and Mueller, Jonas and Erickson, Nick and Li, Mu and Smola, Alex},
  booktitle={8th ICML Workshop on Automated Machine Learning (AutoML)},
  year={2021}
}
```


## AutoGluon for Hyperparameter Optimization

AutoGluon's state-of-the-art tools for hyperparameter optimization, such as ASHA, Hyperband, Bayesian Optimization and BOHB have moved to the stand-alone package [syne-tune](https://github.com/awslabs/syne-tune).

To learn more, checkout our paper ["Model-based Asynchronous Hyperparameter and Neural Architecture Search"](https://arxiv.org/abs/2003.10865) arXiv preprint arXiv:2003.10865 (2020).

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
