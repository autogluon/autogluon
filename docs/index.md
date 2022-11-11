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

AutoML for Tabular, Text, Image, and Multimodal Data

```{button-link} tabular/quick_start.html
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
: Easily deploy to the cloud.

Customizable
: Extensible with custom feature processing, model, and metrics.

## {octicon}`rocket` Quick Examples

:::{dropdown} Tabular Prediction
:animate: fade-in-slide-down
:open:
:color: primary

Predict the `class` column on a data table:

```python
from autogluon.tabular import TabularDataset, TabularPredictor

root = 'https://autogluon.s3.amazonaws.com/datasets/Inc/'
train_data = TabularDataset(root+'train.csv')
predictor = TabularPredictor(label='class').fit(train_data=train_data)

test_data = TabularDataset(root+'test.csv')
predictions = predictor.predict(test_data)
```
:::


:::{dropdown} Text Classification
:animate: fade-in-slide-down
:color: primary

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

:::

:::{dropdown} Image Classification
:animate: fade-in-slide-down
:color: primary

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

:::


:::{admonition} Object Detection
:class: dropdown
:open:

asdfasf
:::

## {octicon}`package` Installation

![](https://img.shields.io/pypi/pyversions/autogluon)
![](https://img.shields.io/pypi/v/autogluon.svg)
![](https://img.shields.io/pypi/dm/autogluon)

AutoGluon supports Linux, MacOS (both Intel and Apple M1), and Windows. 
To install use [pip](https://pip.pypa.io/en/stable/installation/):

```bash
pip install autogluon
```

Check {doc}`./install` for detailed instructions. 


## {octicon}`light-bulb` Solutions to ML problems

::::{grid} 3
:gutter: 3

:::{grid-item-card}  Predicting Columns in a Table
:link: tabular/quick_start.html

Fitting models with tabular datasets
:::

:::{grid-item-card}  Example 1
:link-type: ref
:link: install

A
:::
:::{grid-item-card}  Example 2
C
:::

:::{grid-item-card}  Example 3
D
:::

:::{grid-item-card}  Example 4
E
:::

::::



```{toctree}
:hidden:
:caption: Get Started
:maxdepth: 1

tabular/quick_start
install
```

```{toctree}
:caption: Tabular
:hidden:
:maxdepth: 2

tabular/fit
tabular/predict
tabular/applications
tabular/customization
```

```{toctree}
:caption: API
:hidden:
:maxdepth: 3

api/autogluon.tabular
api/autogluon.core
```
