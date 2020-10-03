

<div align="left">
  <img src="https://user-images.githubusercontent.com/16392542/77208906-224aa500-6aba-11ea-96bd-e81806074030.png" width="350">
</div>

## AutoML Toolkit for Deep Learning

[![Build Status](http://ci.mxnet.io/view/all/job/autogluon/job/master/badge/icon)](http://ci.mxnet.io/view/all/job/autogluon/job/master/)
[![Pypi Version](https://img.shields.io/pypi/v/autogluon.svg)](https://pypi.org/project/autogluon/#history)
![Upload Python Package](https://github.com/awslabs/autogluon/workflows/Upload%20Python%20Package/badge.svg)

AutoGluon automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications.  With just a few lines of code, you can train and deploy high-accuracy deep learning models on tabular, image, and text data.

## Example

```python
# First install package from terminal:
# python3 -m pip install --upgrade pip
# python3 -m pip install --upgrade "mxnet<2.0.0"
# python3 -m pip install autogluon

from autogluon import TabularPrediction as task
train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
predictor = task.fit(train_data=train_data, label='class')
performance = predictor.evaluate(test_data)
```

## Resources

See the [AutoGluon Website](http://autogluon.mxnet.io/index.html) for [documentation](https://autogluon.mxnet.io/api/index.html) and instructions on:
- [Installing AutoGluon](http://autogluon.mxnet.io/index.html#installation)
- [Learning with tabular data](http://autogluon.mxnet.io/tutorials/tabular_prediction/tabular-quickstart.html)
  - [Tips to maximize accuracy](http://autogluon.mxnet.io/tutorials/tabular_prediction/tabular-quickstart.html#maximizing-predictive-performance)
- [Learning with image data](http://autogluon.mxnet.io/tutorials/image_classification/beginner.html)
- [Learning with text data](http://autogluon.mxnet.io/tutorials/text_prediction/beginner.html)
- More advanced topics such as [Neural Architecture Search](http://autogluon.mxnet.io/tutorials/nas/index.html)

### Scientific Publications
- [AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data](https://arxiv.org/pdf/2003.06505.pdf) (*Arxiv*, 2020)

### Articles
- [AutoGluon for tabular data: 3 lines of code to achieve top 1% in Kaggle competitions](https://aws.amazon.com/blogs/opensource/machine-learning-with-autogluon-an-open-source-automl-library/) (*AWS Open Source Blog*, Mar 2020)
- [Accurate image classification in 3 lines of code with AutoGluon](https://medium.com/@zhanghang0704/image-classification-on-kaggle-using-autogluon-fc896e74d7e8) (*Medium*, Feb 2020)
- [AutoGluon overview & example applications](https://towardsdatascience.com/autogluon-deep-learning-automl-5cdb4e2388ec?source=friends_link&sk=e3d17d06880ac714e47f07f39178fdf2) (*Towards Data Science*, Dec 2019)

### Hands-on Tutorials
- [From HPO to NAS: Automated Deep Learning (CVPR 2020)](http://hangzhang.org/CVPR2020/)
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

Also have a look at our paper ["Model-based Asynchronous Hyperparameter and Neural Architecture Search"](https://arxiv.org/abs/2003.10865) arXiv preprint arXiv:2003.10865 (2020).

```bibtex
@article{abohb,
  title={Model-based Asynchronous Hyperparameter and Neural Architecture Search},
  author={Klein, Aaron and Tiao, Louis and Lienart, Thibaut and Archambeau, Cedric and Seeger, Matthias}
  journal={arXiv preprint arXiv:2003.10865},
  year={2020}
}
```

## License

This library is licensed under the Apache 2.0 License.

## Contributing to AutoGluon

We are actively accepting code contributions to the AutoGluon project. If you are interested in contributing to AutoGluon, please read the [Contributing Guide](https://github.com/awslabs/autogluon/blob/master/CONTRIBUTING.md) to get started.
