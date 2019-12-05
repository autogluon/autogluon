# Object Detection - Quick Start
:label:`sec_object_detection_quick`

Besides Image classification, object detection is another important task in computer vision. In this tutorial, we share a brief example to go through the steps that how to use AutoGluon for object detection.

Our goal is to detect motorbike in images by YOLO3 model. A tiny dataset is collected from VOC dataset, which only contains the motorbike category. The model pretrained on COCO dataset is used to do fine-tuning our small dataset. With the help of AutoGluon, we are able to try many models with different hyperparameters automatically, and return the best one as our final model. It is better to go over the tutorial about :ref:`sec_imgquick` first to learn how to use AutoGluon.

Let's start with importing autogluon and ObjectDetection module for our task: 
```{.python .input}
import autogluon as ag
from autogluon import ObjectDetection as task
```

## Tiny_motorbike Dataset
We collect a toy dataset only for detecting motorbikes in images. From VOC dataset, 120 images are randomly selected as our training dataset, 50 images for validation and another 50 ones for testing. This tiny dataset follows the same format as VOC. 

Using the commands below, we can download this dataset, which is only 23M. The variable `root` specifies the path to store this dataset. The name of unzipped folder is called `tiny_motorbike`.

```{.python .input}
filename = ag.download('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')
ag.unzip(filename)
```

When we retrieve the dataset, we can create dataset instance with its name and path.
```{.python .input}
dataset = task.Dataset('tiny_motorbike', root='./tiny_motorbike')
```

## Fit Models by AutoGluon
In this section, we demonstrate how to apply AutoGluon to fit our detection models. We use mobilenet as backbone for YOLO3 model. Two different learning rates are used to fine-tune the network. The best model is the one that obtains the best performance on validation dataset. We can also try more networks and hyperparameters to have a larger searching space. 

In each experiment (one trial in our searching space), we train the model for 30 epoches. Let's start playing around with AutoGluon.  


```{.python .input}
epochs = 30
detector = task.fit(dataset,
                    epochs=epochs,
                    ngpus_per_trial=1)
```

After fitting, AutoGluon automatically returns the best model among all models in the searching space. From the output, we know the best model is the one trained with the second learning rate. Now, let's see how well the returned model performed on test dataset by simply calling detector.evaluate().

```{.python .input}
test_map = detector.evaluate(dataset) # it only evaluates on test dataset.
print("mAP on test dataset: {}".format(test_map[1][1]))
```

The mAP is not bad after just 30 epochs. Let's see one visualization result. We randomly select an image from test dataset, and show predicted bbox and probability over the origin image.  

```{.python .input}
image_path = './tiny_motorbike/VOC2007/JPEGImages/000467.jpg'
ind, prob, loc = detector.predict(image_path)
```
