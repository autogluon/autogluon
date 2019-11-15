# Object Detection - Quick Start

Besides Image classification, objecgt detection is another important task in computer vision. In this tutorial, we will share a brief example to go through steps that how to use AutoGluon for object detection.

Our goal is to detect motorbike in images by YOLO3 model. A tiny dataset is collected from VOC dataset, which only contains motorbike category. The model that is pretained on COCO dataset is used to do finetuning on our small dataset. With the help of AutoGluon, we are able to try many models with different hyper-parameters automatically, and return the best one as our final model. It is better to go over the #TODO: imagenet classification

Let's start with importing autogluon and ObjectDetection module for our task 
```{.python .input}
import autogluon as ag
from autogluon import ObjectDetection as task
```

## Tiny_motorbike dataset
We collect a toy dataset only for detecting motorbikes in images. From VOC datset, 120 images are selected as our training dataset, 50 images for validation and another 50 ones for testing. This tiny dataset follows the same format as VOC. 

The dataset, which is only 23M, could be downloaded by commands below. The variable `root` specifies the path to store this dataset. The name of unzipped folder is called `tiny_motorbike`.

```{.python .input}
import os
root = '/home/ubuntu/'
filename = ag.download('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip',
                        path=root)
ag.unzip(filename, root=root)
```

```{.python .input}
dataset = task.Dataset("tiny_motorbike", root=os.path.join(root, "tiny_motorbike"))
```

```{.python .input}
time_limits = 3*60 # 3mins
epochs = 30
classifier = task.fit(dataset,
                      net=Categorical('mobilenet1.0'),
                      lr=Categorical(5e-4, 1e-4),
                      time_limits=time_limits,
                      epochs=epochs,
                      ngpus_per_trial=1)
```

```{.python .input}
test_acc = classifier.evaluate(dataset)
print("mAP on test dataset: {}".format(test_acc[1][1]))
```

```{.python .input}
ind, prob, loc = classifier.predict(image_path)
```

