# AutoMM Detection - Inference Quick Start with Pretrained VFNet
:label:`sec_automm_detection_infer_tiny_motorbike`

In this section, we show an quick-start example to run inference on a single image and visualize the detections.  
The model we use is the VFNet pretrained on COCO dataset.

## Prepare data
```{.python .input}
import os
import time
```

Download the example image:
```{.python .input}
!wget https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg -q -O input.jpg
```

[//]: # (Now let's download the dataset)


[//]: # (download_dir = "./tiny_motorbike_coco"  # specify a target download dir to store this dataset)

[//]: # ()
[//]: # (load_zip.unzip&#40;zip_file, unzip_dir=download_dir&#41;)

[//]: # (data_dir = os.path.join&#40;download_dir, "tiny_motorbike"&#41;)

[//]: # (train_path = os.path.join&#40;data_dir, "Annotations", "train_cocoformat.json"&#41;)

[//]: # (test_path = os.path.join&#40;data_dir, "Annotations", "test_cocoformat.json"&#41;)

[//]: # (```)

## Creating the `MultiModalPredictor`
To start, import MultiModalPredictor:
```{.python .input}
from autogluon.multimodal import MultiModalPredictor
```

AutoMM detection requires `mmcv-full` and `mmdet` packages. Please make sure `mmcv-full` and `mmdet` are installed:
```{.python .input}
!mim install mmcv-full
!pip install mmdet
```

### Use a pretrained model
You can download a pretrained model and construct a predictor with it. 
In this example, we use the VFNet with ResNext as backbone and Feature Pyramid Network (FPN) as neck.

```{.python .input}
checkpoint_name = "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco"
num_gpus = 1  # set to -1 to use all GPUs if available
```
You can also use other model by setting `checkpoint_name` to other names. 
Please refer to :ref: `selecting_models` for details about model selection.

As before, we create the MultiModalPredictor with selected checkpoint name and number of GPUs.
We also need to specify the `problem_type` to `"object_detection"`.

```{.python .input}
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
)
```

## Setting up data

For COCO format data, we need to provide the path for the data split used for inference.

```{.python .input}
test_image = "./input.jpg"
```

## Running inference
To run inference, perform:

```{.python .input}
pred = predictor.predict({"image": [test_image]}, as_pandas=True)
print(pred)
```

The output `pred` is a `pandas` `DataFrame` that has two columns, `image` and `bboxes`, where
- in `image`, each row contains the image path
- in `bboxes`, each row is a list of dictionaries, each one representing a bounding box: 
  - `{"class": <predicted_class_name>, "bbox": [x1, y1, x2, y2], "score": <confidence_score>}`


## Visualizing Results
To run visualizations, ensure that you have `opencv` installed. If you haven't already, install `opencv` by running 
```{.python .input}
!pip install opencv-python
```

To visualize the detection bounding boxes, run the following:
```{.python .input}
from autogluon.multimodal.utils import from_coco_or_voc, visualize_detection
from matplotlib import pyplot as plt

conf_threshold = 0.4  # Specify a confidence threshold to filter out unwanted boxes
visualization_result_dir = "./"  # Use the pwd as result dir

visualized = visualize_detection(
    pred=pred,
    detection_classes=predictor.get_predictor_classes(),
    conf_threshold=conf_threshold,
    visualization_result_dir=visualization_result_dir,
)

plt.imshow(visualized[0][:, : ,::-1])  # shows the first image with bounding box
```

### Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

### Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.

### Citation
```
@inproceedings{zhang2021varifocalnet,
  title={Varifocalnet: An iou-aware dense object detector},
  author={Zhang, Haoyang and Wang, Ying and Dayoub, Feras and Sunderhauf, Niko},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8514--8523},
  year={2021}
}
```