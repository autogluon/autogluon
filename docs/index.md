---
sd_hide_title: true
hide-toc: true
---

# AutoGluon

::::::{div} landing-title
:style: "padding: 0.1rem 0.5rem 0.6rem 0; background-image: linear-gradient(315deg, #438ff9 0%, #3977B9 74%); clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem)); -webkit-clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem));"

::::{grid}
:reverse:
:gutter: 2 3 3 3
:margin: 4 4 1 2

:::{grid-item}
:columns: 12 4 4 4

```{image} ./_static/autogluon-s.png
:width: 200px
:class: sd-m-auto sd-animate-grow50-rot20
```
:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-text-white sd-fs-3

AutoML for Tabular, Text, Image, Multimodal, and Time Series Data

```{button-link} tutorials/tabular/tabular_quick_start.html
:outline:
:color: white
:class: sd-px-4 sd-fs-5

Get Started
```

:::
::::

::::::

Quick Prototyping 
: Build machine learning solutions on raw data in a few lines of code.

State-of-the-art Techniques
: Automatically utilize SOTA models without expert knowledge.

Easy to Deploy
: Move from experimentation to production with cloud predictors and pre-built containers.

Customizable
: Extensible with custom feature processing, models, and metrics.

## {octicon}`rocket` Quick Examples

:::{dropdown} Tabular Prediction
:animate: fade-in-slide-down
:open:
:color: primary

Predict the `class` column on a data table:

```python
from autogluon.tabular import TabularDataset, TabularPredictor

data_root = 'https://autogluon.s3.amazonaws.com/datasets/Inc/'
train_data = TabularDataset(data_root + 'train.csv')
test_data = TabularDataset(data_root + 'test.csv')

predictor = TabularPredictor(label='class').fit(train_data=train_data)
predictions = predictor.predict(test_data)
```
:::


:::{dropdown} Text Classification
:animate: fade-in-slide-down
:color: primary

Predict sentiment of movie reviews:

```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.core.utils.loaders import load_pd

data_root = 'https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/'
train_data = load_pd.load(data_root + 'train.parquet')
test_data = load_pd.load(data_root + 'dev.parquet')

predictor = MultiModalPredictor(label='label').fit(train_data=train_data)
predictions = predictor.predict(test_data)
```
:::

:::{dropdown} Image Classification
:animate: fade-in-slide-down
:color: primary

Predict clothing article types:

```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.vision import ImageDataset

data_zip = 'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip'
train_data, _, test_data = ImageDataset.from_folders(data_zip)

predictor = MultiModalPredictor(label='label').fit(train_data=train_data)
predictions = predictor.predict(test_data)
```
:::


## {octicon}`package` Installation

![](https://img.shields.io/pypi/pyversions/autogluon)
![](https://img.shields.io/pypi/v/autogluon.svg)
![](https://img.shields.io/pypi/dm/autogluon)

Install AutoGluon using [pip](https://pip.pypa.io/en/stable/installation/):

```bash
python -m pip install autogluon
```

AutoGluon supports Linux, MacOS, and Windows. See {doc}`./install` for detailed instructions. 


```{toctree}
:hidden:
:caption: Get Started
:maxdepth: 1

install
tutorials/tabular/tabular_quick_start
tutorials/multimodal/multimodal_quick_start
tutorials/timeseries/timeseries_quick_start
```

```{toctree}
:caption: API
:hidden:
:maxdepth: 3

api/autogluon.tabular
api/autogluon.core
```
