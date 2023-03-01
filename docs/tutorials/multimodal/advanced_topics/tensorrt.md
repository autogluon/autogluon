# Faster Prediction with TensorRT
:label:`sec_automm_tensorrt`

AutoMM is a deep learning "model zoo" of model zoos. It can automatically build deep learning models that are suitable for multimodal datasets. You will only need to convert the data into the multimodal dataframe format
and AutoMM can predict the values of one column conditioned on the features from the other columns including images, text, and tabular data.


```{.python .input}
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Dataset

For demonstration, we use a simplified and subsampled version of [PetFinder dataset](https://www.kaggle.com/c/petfinder-adoption-prediction). The task is to predict the animals' adoption rates based on their adoption profile information. In this simplified version, the adoption speed is grouped into two categories: 0 (slow) and 1 (fast).

To get started, let's download and prepare the dataset.


```{.python .input}
download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_for_tutorial.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

Next, we will load the CSV files.


```{.python .input}
import pandas as pd
dataset_path = download_dir + '/petfinder_for_tutorial'
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
label_col = 'AdoptionSpeed'
```

We need to expand the image paths to load them in training.


```{.python .input}
image_col = 'Images'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0]) # Use the first image for a quick tutorial
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])


def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
```

Each animal's adoption profile includes pictures, a text description, and various tabular features such as age, breed, name, color, and more. Refer to :ref:`sec_automm_multimodal_beginner` for visualization of an example row of the dataset.

## Training
Now let's fit the predictor with the training data. Here we set a tight time budget for a quick demo.

```{.python .input}
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(label=label_col).fit(
    train_data=train_data,
    time_limit=120, # seconds
)
```
Under the hood, AutoMM automatically infers the problem type (classification or regression), detects the data modalities, selects the related models from the multimodal model pools, and trains the selected models. If multiple backbones are available, AutoMM appends a late-fusion model (MLP or transformer) on top of them.

## Prediction with vanilla module
Given a multimodal dataframe without the label column, we can predict the labels.

```{.python .input}
for batch_size in [2, 4, 8]:
    sample = test_data.head(batch_size)
    for _ in range(3):
        tic = time.time()
        y_pred = predictor.predict(sample)
        print(f"elapsed (vanilla): {(time.time()-tic)*1000:.1f} ms (batch_size={batch_size})")
```

## Prediction with TensorRT module
First, we need to export the module to ONNX, in order to use TensorrtExecutionProvider in onnxruntime for prediction.

```{.python .input}
sample = test_data.head(2)
trt_module = predictor.export_tensorrt(data=sample)
```

The exported OnnxModule can be a drop-in replacement of torch.nn.Module. Therefore, we can replace the internal neural network module directly.

```{.python .input}
predictor._model = trt_module
```

Then, we can perform prediction or extract embeddings as usual. To verify dynamic shape support, we can predict with varying batch sizes

```{.python .input}
for batch_size in [2, 4, 8]:
    sample = test_data.head(batch_size)
    for _ in range(3):
        tic = time.time()
        y_pred_trt = predictor.predict(sample)
        print(f"elapsed (tensorrt): {(time.time()-tic)*1000:.1f} ms (batch_size={batch_size})")
```

To verify the correctness of the prediction results, we can compare the results side-by-side.

Let's print the expected results first.

```{.python .input}
y_pred
```

Then the results from TensorRT.

```{.python .input}
y_pred_trt
```

We can safely assume these results are relatively close for most of the cases.

```{.python .input}
np.testing.assert_allclose(y_pred, y_pred_trt, rtol=1e-3)
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
