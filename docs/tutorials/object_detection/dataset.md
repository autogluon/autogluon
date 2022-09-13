# Object Detection - Prepare Dataset for Object Detector

:label:`sec_object_detection_dataset`

Preparing dataset for object detection is slightly difference and more difficult than image prediction.

Our goal in this tutorial is to introduce the simplest methods to initiate or load a object detection dataset for `autogluon.vision.ObjectDetector`.

There are generally two ways to load a dataset for ObjectDetector:

- Load an existing object detection dataset, in VOC or COCO formats, downloaded or exported by other labeling tools.

- Manually convert raw annotations in any format, knowing this you will be able to deal with arbitrary dataset format.

```{.python .input}
%matplotlib inline
import autogluon.core as ag
from autogluon.vision import ObjectDetector
```

## Load an existing object detection dataset

Pascal VOC and MS COCO are two most popular data format for object detection. Most public available object detection datasets follow either one of these two formats. In this tutorial we will not touch the details. You may view the original introduction for [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](https://cocodataset.org/#home).

To distinguish these two formats, you can either refer to the labeling tool or check the folder structure. Usually annotations in VOC format are individual `xml` files, while COCO format use a single `json` file to store all annotations.

```{.python .input}
url = 'https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip'
dataset_train = ObjectDetector.Dataset.from_voc(url, splits='trainval')
# or load from coco format, skip as it's too big to download
# dataset_train = ObjectDetector.Dataset.from_coco(annotation_json_file, root='/path/to/root')
```

## Manually convert any format to autogluon object detector dataset

We will walk you through by creating a dataset manually to help you understand the meaning of underlying data, this does not mean you have to do so. We highly recommend you to use a handy labeling tool for object detection if you want to create one by your own. Labeling bounding boxes are time consuming so a nice UI/UX design will significantly reduce the trouble.

In the following section, we will use a single image and add annotations manually for all three major objects.

```{.python .input}
ag.utils.download('https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg', path='dog.jpg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img = mpimg.imread('dog.jpg')
imgplot = plt.imshow(img)
plt.grid()
plt.show()
```

With the grid on, we can roughly annotate this image like this:

```{.python .input}
import pandas as pd

class NaiveDetectionGT:
    def __init__(self, image):
        self._objects = []
        self.image = image
        img = mpimg.imread('dog.jpg')
        self.w = img.shape[1]
        self.h = img.shape[0]

    def add_object(self, name, xmin, ymin, xmax, ymax, difficult=0):
        self._objects.append({'image': self.image, 'class': name,
                              'xmin': xmin / self.w, 'ymin': ymin / self.h,
                              'xmax': xmax / self.w, 'ymax': ymax / self.h, 'difficult': difficult})

    @property
    def df(self):
        return pd.DataFrame(self._objects)

gt = NaiveDetectionGT('dog.jpg')
gt.add_object('dog', 140, 220, 300, 540)
gt.add_object('bicycle', 120, 140, 580, 420)
gt.add_object('car', 460, 70, 680, 170)
df = gt.df
df
```

The `df` is a valid dataset and can be used by `ObjectDetector.fit` function. Internally it will be converted to object detection dataset, or you can manually convert it.

```{.python .input}
dataset = ObjectDetector.Dataset(df, classes=df['class'].unique().tolist())
dataset.show_images(nsample=1, ncol=1)
```

Congratulations, you can now proceed to :ref:`sec_object_detection_quick` to start training the `ObjectDetector`.
