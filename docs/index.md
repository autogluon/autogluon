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

AutoML for Image, Text, Time Series, and Tabular Data

```{button-link} tutorials/tabular/tabular-quick-start.html
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

:::{dropdown} Tabular
:animate: fade-in-slide-down
:open:
:color: primary

Predict the `class` column in a data table:

```python
from autogluon.tabular import TabularDataset, TabularPredictor

data_root = 'https://autogluon.s3.amazonaws.com/datasets/Inc/'
train_data = TabularDataset(data_root + 'train.csv')
test_data = TabularDataset(data_root + 'test.csv')

predictor = TabularPredictor(label='class').fit(train_data=train_data)
predictions = predictor.predict(test_data)
```
:::


::::{dropdown} Multimodal
:animate: fade-in-slide-down
:color: primary

:::{tab} Text Classification
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

:::{tab} Image Classification

```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.vision import ImageDataset

data_zip = 'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip'
train_data, _, test_data = ImageDataset.from_folders(data_zip)

predictor = MultiModalPredictor(label='label').fit(train_data=train_data)
predictions = predictor.predict(test_data)
```
:::

:::{tab} Named Entity Recognition
```python
from autogluon.multimodal import MultiModalPredictor
from autogluon.core.utils.loaders import load_pd

data_root = 'https://automl-mm-bench.s3.amazonaws.com/ner/mit-movies/'
train_data = load_pd.load(data_root + 'train.csv')
test_data = load_pd.load(data_root + 'test.csv')

predictor = MultiModalPredictor(problem_type="ner", label="entity_annotations")

predictor.fit(train_data)
predictor.evaluate(test_data)

sentence = "Game of Thrones is an American fantasy drama television series created" +
           "by David Benioff"
prediction = predictor.predict({ 'text_snippet': [sentence]})
```
:::

:::{tab} Object Detection
```python
# !mim install mmcv-full
# !pip install mmdet
from autogluon.multimodal import MultiModalPredictor

data_zip = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/" +
           "tiny_motorbike_coco.zip"
# Unzip dataset

train_path = "./Annotations/trainval_cocoformat.json"
test_path = "./Annotations/test_cocoformat.json"

predictor = MultiModalPredictor(
  problem_type="object_detection",
  sample_data_path=train_path
)

predictor.fit(train_path)
predictor.evaluate(test_path)

pred = predictor.predict({"image": [test_image]})
```
:::

:::{tab} Matching
```python
from autogluon.multimodal import MultiModalPredictor, utils
import ir_datasets
import pandas as pd

dataset = ir_datasets.load("beir/fiqa/dev")
docs_df = pd.DataFrame(dataset.docs_iter()).set_index("doc_id")

predictor = MultiModalPredictor(problem_type="text_similarity")

doc_embedding = predictor.extract_embedding(docs_df)
q_embedding = predictor.extract_embedding([
  "what happened when the dot com bubble burst?"
])

similarity = utils.compute_semantic_similarity(q_embedding, doc_embedding)
```
:::
::::


:::{dropdown} Time Series
:animate: fade-in-slide-down
:color: primary

Forecast future values of time series:

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

data = TimeSeriesDataFrame('https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv')

predictor = TimeSeriesPredictor(target='target', prediction_length=48).fit(data)
predictions = predictor.predict(data)
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
---
caption: Get Started
maxdepth: 1
hidden:
---

install
Tabular Quick Start <tutorials/tabular/tabular-quick-start>
Multimodal Quick Start <tutorials/multimodal/multimodal-quick-start>
Time Series Quick Start <tutorials/timeseries/forecasting-quick-start>
```

```{toctree}
---
caption: Tutorials
maxdepth: 3
hidden:
---

tutorials/tabular/index
tutorials/multimodal/index
tutorials/timeseries/index
```

```{toctree}
---
caption: Resources
maxdepth: 1
hidden:
---

Cheat Sheets <cheatsheet.rst>
Versions <versions.rst>
Tabular FAQ <tutorials/tabular/tabular-faq.md>
Multimodal FAQ <tutorials/multimodal/multimodal-faq.md>
Time Series FAQ <tutorials/timeseries/forecasting-faq.md>
```


```{toctree}
---
caption: API
maxdepth: 1
hidden:
---

api/_autogen/autogluon.tabular.TabularPredictor
api/_autogen/autogluon.tabular.TabularDataset
api/_autogen/autogluon.multimodal.MultiModalPredictor
api/_autogen/autogluon.timeseries.TimeSeriesDataFrame
api/_autogen/autogluon.timeseries.TimeSeriesPredictor
api/_autogen/autogluon.common.features.feature_metadata.FeatureMetadata
```
