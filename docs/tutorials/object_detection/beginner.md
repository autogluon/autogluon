# Object Detection - Quick Start
:label:`sec_object_detection_quick`

**Note**: AutoGluon ObjectDetector will be deprecated in v0.7. Please try our [AutoGluon MultiModalPredictor](https://auto.gluon.ai/stable/tutorials/multimodal/object_detection/quick_start/quick_start_coco.html) for more functionalities and better support for your object detection  need.

Object detection is the process of identifying and localizing objects in an image and is an important task in computer vision. Follow this tutorial to learn how to use AutoGluon for object detection.

**Tip**: If you are new to AutoGluon, review :ref:`sec_imgquick` first to learn the basics of the AutoGluon API.

Our goal is to detect motorbike in images by [YOLOv3 model](https://pjreddie.com/media/files/papers/YOLOv3.pdf). A tiny dataset is collected from VOC dataset, which only contains the motorbike category. The model pretrained on the COCO dataset is used to fine-tune our small dataset. With the help of AutoGluon, we are able to try many models with different hyperparameters automatically, and return the best one as our final model.

To start, import ObjectDetector:

```{.python .input}
from autogluon.vision import ObjectDetector
```

## Tiny_motorbike Dataset
We collect a toy dataset for detecting motorbikes in images. From the VOC dataset, images are randomly selected for training, validation, and testing - 120 images for training, 50 images for validation, and 50 for testing. This tiny dataset follows the same format as VOC.

Using the commands below, we can download this dataset, which is only 23M. The name of unzipped folder is called `tiny_motorbike`. Anyway, the task dataset helper can perform the download and extraction automatically, and load the dataset according to the detection formats.

```{.python .input}
url = 'https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip'
dataset_train = ObjectDetector.Dataset.from_voc(url, splits='trainval')
```

## Fit Models by AutoGluon
In this section, we demonstrate how to apply AutoGluon to fit our detection models. We use mobilenet as the backbone for the YOLOv3 model. Two different learning rates are used to fine-tune the network. The best model is the one that obtains the best performance on the validation dataset. You can also try using more networks and hyperparameters to create a larger searching space.

We `fit` a classifier using AutoGluon as follows. In each experiment (one trial in our searching space), we train the model for 5 epochs to avoid bursting our tutorial runtime.

```{.python .input}
time_limit = 60*30  # at most 0.5 hour
detector = ObjectDetector()
hyperparameters = {'epochs': 5, 'batch_size': 8}
hyperparameter_tune_kwargs={'num_trials': 2}
detector.fit(dataset_train, time_limit=time_limit, hyperparameters=hyperparameters, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs)

```

Note that `num_trials=2` above is only used to speed up the tutorial. In normal
practice, it is common to only use `time_limit` and drop `num_trials`. Also note
that hyperparameter tuning defaults to random search.

After fitting, AutoGluon automatically returns the best model among all models in the searching space. From the output, we know the best model is the one trained with the second learning rate. To see how well the returned model performed on test dataset, call detector.evaluate().

```{.python .input}
dataset_test = ObjectDetector.Dataset.from_voc(url, splits='test')

test_map = detector.evaluate(dataset_test)
print("mAP on test dataset: {}".format(test_map[1][-1]))
```

Below, we randomly select an image from test dataset and show the predicted class, box and probability over the origin image, stored in `predict_class`, `predict_rois` and `predict_score` columns, respectively. You can interpret `predict_rois` as a dict of (`xmin`, `ymin`, `xmax`, `ymax`) proportional to original image size.

```{.python .input}
image_path = dataset_test.iloc[0]['image']
result = detector.predict(image_path)
print(result)
```

Prediction with multiple images is permitted:
```{.python .input}
bulk_result = detector.predict(dataset_test)
print(bulk_result)
```

We can also save the trained model, and use it later.

:::warning

`ObjectDetector.load()` used `pickle` module implicitly, which is known to be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Never load data that could have come from an untrusted source, or that could have been tampered with. **Only load data you trust.**

:::

```{.python .input}
savefile = 'detector.ag'
detector.save(savefile)
new_detector = ObjectDetector.load(savefile)
```
