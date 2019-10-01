# AutoGluon: AutoML Toolkit with MXNet Gluon

## Why deep learning? 

In just the past five years, deep learning has taken the world by surprise, driving rapid progress in fields as diverse as computer vision, natural language processing, automatic speech recognition, reinforcement learning, and statistical modeling. With these advances in hand, we can now build cars that drive themselves (with increasing autonomy), smart reply systems that anticipate mundane replies, helping people dig out from mountains of email, and software agents that dominate the world’s best humans at board games like Go, a feat once deemed to be decades away. Already, these tools are exerting a widening impact, changing the way movies are made, diseases are diagnosed, and playing a growing role in basic sciences – from astrophysics to biology
-*`Dive into Deep Learning` book by Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola*

## Why automated machine learning?

It is obvious that deep learning is powerful. Everyone needs to learn it. However, mastering different pieces in deep learning is not that easy. In general, large amounts of data are available everywhere, but it lacks of domain or machine learning experts who can advise the development of the deep learning systems. 
To be more specific, automatic model generation, tuning, selection, and performance optimization is a common problem for many scientists and practitioners alike. It is time consuming to find good models. Despite meaningful progress in terms of algorithms for automatic model inference they have not found widespread practical acceptance due to their difficulty to use them in the real world. 
Automated machine learning (AutoML) provides the methods and processes to make machine learning available for non-machine learning experts, thus to improve the efficiency and accelerate research of machine learning.

## Why AutoGluon?

We present `AutoGluon` which is an AutoML toolkit with MXNet Gluon that enables easy-to-use and easy-to-extend AutoML with a focus on deep learning, and making AutoML deploy in real-world applications.
`AutoGluon` supports AutoML functionalities including hyper-parameter optimization search algorithms, cell based architecture search, early stopping mechanisms for a wide range of CV and NLP applications.
With just a single call to `AutoGluon`'s `fit` function, AutoGluon will automatically train many models and thousands of different hyperparameter configurations regarding to the training process and return the best model.
Besides, you could easily specify for greater control over the training process such as providing the time limits for the training procedures and how many computation resource you want each training run leverage. 
In this hackathon, we will provide an image classification example to show the usage of `AutoGluon`'s main APIs including `fit`, and how to easily use your own dataset and control the searching process to produce competitive results within 5 hours in Kaggle image classification competition as well as achieve the state-of-the-art image classification results on CIFAR10 (one of the benchmark image classification datasets).

## Table of Contents

```toc
:maxdepth: 1

install
```

```toc
:numbered:
:maxdepth: 2

tutorials/index
```

```toc
:maxdepth: 2

api/index
```

## Next steps

- For new users: [60-minute Gluon crash course](https://beta.mxnet.io/guide/getting-started/crash-course/index.html).
- For experienced users: [MXNet Guides](http://mxnet.apache.org/versions/master/tutorials/index.html#python-tutorials).
- For advanced users: [MXNet API](https://beta.mxnet.io/api/index.html).
