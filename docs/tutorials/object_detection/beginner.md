# Object Detection - Quick Start
:label:`sec_object_detection_quick`

Object detection is the process of identifying and localizing objects in an image and is an important task in computer vision. Follow this tutorial to learn how to use AutoGluon for object detection.

**Tip**: If you are new to AutoGluon, review :ref:`sec_imgquick` first to learn the basics of the AutoGluon API.

Our goal is to detect motorbike in images by [YOLO3 model](https://pjreddie.com/media/files/papers/YOLOv3.pdf). A tiny dataset is collected from VOC dataset, which only contains the motorbike category. The model pretrained on the COCO dataset is used to fine-tune our small dataset. With the help of AutoGluon, we are able to try many models with different hyperparameters automatically, and return the best one as our final model. 

To start, import autogluon and ObjectDetection module as your task: 

```{.python .input}
import autogluon as ag
from autogluon import ObjectDetection as task
```

## Tiny_motorbike Dataset
We collect a toy dataset for detecting motorbikes in images. From the VOC dataset, images are randomly selected for training, validation, and testing - 120 images for training, 50 images for validation, and 50 for testing. This tiny dataset follows the same format as VOC. 

Using the commands below, we can download this dataset, which is only 23M. The variable `root` specifies the path to store the dataset in. The name of unzipped folder is called `tiny_motorbike`.

```{.python .input}
root = './'
filename_zip = ag.download('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip',
                        path=root)
filename = ag.unzip(filename_zip, root=root)
```

When we retrieve the dataset, we can create a dataset instance with its path and classes if it is a custom dataset.

```{.python .input}
import os
data_root = os.path.join(root, filename)
dataset_train = task.Dataset(data_root, classes=('motorbike',))
```

## Fit Models by AutoGluon
In this section, we demonstrate how to apply AutoGluon to fit our detection models. We use mobilenet as the backbone for the YOLO3 model. Two different learning rates are used to fine-tune the network. The best model is the one that obtains the best performance on the validation dataset. You can also try using more networks and hyperparameters to create a larger searching space. 

We `fit` a classifier using AutoGluon as follows. In each experiment (one trial in our searching space), we train the model for 30 epoches. 

```{.python .input}
time_limits = 5*60*60  # 5 hours
epochs = 30
detector = task.fit(dataset_train,
                    num_trials=2,
                    epochs=epochs,
                    lr=ag.Categorical(5e-4, 1e-4),
                    ngpus_per_trial=1,
                    time_limits=time_limits)
```

Note that `num_trials=2` above is only used to speed up the tutorial. In normal
practice, it is common to only use `time_limits` and drop `num_trials`.

After fitting, AutoGluon automatically returns the best model among all models in the searching space. From the output, we know the best model is the one trained with the second learning rate. To see how well the returned model performed on test dataset, call detector.evaluate().

```{.python .input}
dataset_test = task.Dataset(data_root, index_file_name='test', classes=('motorbike',))

test_map = detector.evaluate(dataset_test)
print("mAP on test dataset: {}".format(test_map[1][1]))
```

Below, we randomly select an image from test dataset and show the predicted box and probability over the origin image.  

```{.python .input}
image = '000467.jpg'
image_path = os.path.join(data_root, 'JPEGImages', image)

ind, prob, loc = detector.predict(image_path)
```

We can also save the trained model, and use it later. 
```{.python .input}
savefile = 'model.pkl'
detector.save(savefile)

from autogluon import Detector
new_detector = Detector.load(savefile)
```

