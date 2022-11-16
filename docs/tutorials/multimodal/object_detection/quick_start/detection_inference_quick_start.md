# AutoMM Detection - Inference Quick Start with Pretrained YOLOv3
:label:`sec_automm_detection_infer_tiny_motorbike`

In this section, we show an quick-start example to run inference on a single image and visualize the detections.  
The model we use is the YOLOv3 pretrained on COCO dataset.

AutoMM detection requires `mmcv-full` and `mmdet` packages. Please make sure `mmcv-full` and `mmdet` are installed:
```python .input
!mim install mmcv-full
!pip install mmdet
```

## Prepare data
```python .input
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)

import os
import time
```

Download the example image:
```python .input
from autogluon.multimodal import download
```
```python .input
from autogluon.multimodal import download
image_url = "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
test_image = download(image_url)
```

## Creating the `MultiModalPredictor`
To start, import MultiModalPredictor:
```python .input
from autogluon.multimodal import MultiModalPredictor
```

### Use a pretrained model
You can download a pretrained model and construct a predictor with it. 
In this example, we use the YOLOv3 with MobileNet backbone.

```python .input
checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
num_gpus = 1  # set to -1 to use all GPUs if available
```
You can also use other model by setting `checkpoint_name` to other names.

As before, we create the MultiModalPredictor with selected checkpoint name and number of GPUs.
We also need to specify the `problem_type` to `"object_detection"`.

```python .input
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
)
```

## Running inference
To run inference, perform:

```python .input
pred = predictor.predict({"image": [test_image]}, as_pandas=True)
print(pred)
```

The output `pred` is a `pandas` `DataFrame` that has two columns, `image` and `bboxes`, where
- in `image`, each row contains the image path
- in `bboxes`, each row is a list of dictionaries, each one representing a bounding box: 
  - `{"class": <predicted_class_name>, "bbox": [x1, y1, x2, y2], "score": <confidence_score>}`


## Visualizing Results
To run visualizations, ensure that you have `opencv` installed. If you haven't already, install `opencv` by running 
```python .input
!pip install opencv-python
```

To visualize the detection bounding boxes, run the following:
```python .input
from autogluon.multimodal.utils import visualize_detection

conf_threshold = 0.4  # Specify a confidence threshold to filter out unwanted boxes
visualization_result_dir = "./"  # Use the pwd as result dir

visualized = visualize_detection(
    pred=pred,
    detection_classes=predictor.get_predictor_classes(),
    conf_threshold=conf_threshold,
    visualization_result_dir=visualization_result_dir,
)

from PIL import Image
from IPython.display import display
img = Image.fromarray(visualized[0][:, :, ::-1], 'RGB')
display(img)
```

### Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

### Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.

### Citation
```
@misc{redmon2018yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Joseph Redmon and Ali Farhadi},
    year={2018},
    eprint={1804.02767},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```