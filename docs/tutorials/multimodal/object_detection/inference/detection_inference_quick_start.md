# AutoMM Detection - Inference with Pretrained VFNet on Tiny Motorbike Dataset
:label:`sec_automm_detection_infer_tiny_motorbike`

In this section, we show an quick-start example to run inference on a small dataset (Tiny Motorbick) that is in COCO Format. 
The model we use is the VFNet pretrained on COCO dataset.

AutoMM detection requires `mmcv-full` and `mmdet` packages. Please make sure `mmcv-full` and `mmdet` are installed:
```{.python}
!mim install mmcv-full
!pip install mmdet
```

## Prepare data
```{.python}
import os
import time

from autogluon.core.utils.loaders import load_zip
```

The data file stored in the cloud is located at:
```{.python}
zip_file = "s3://automl-mm-bench/object_detection_dataset/tiny_motorbike_coco.zip"
```

Now let's download the dataset
```{.python}
download_dir = "./tiny_motorbike_coco"  # specify a target download dir to store this dataset

load_zip.unzip(zip_file, unzip_dir=download_dir)
data_dir = os.path.join(download_dir, "tiny_motorbike")
train_path = os.path.join(data_dir, "Annotations", "train_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
```

## Creating the `MultiModalPredictor`
To start, import MultiModalPredictor:
```{.python}
from autogluon.multimodal import MultiModalPredictor
```
### Use a pretrained model
You can download a pretrained model and construct a predictor with it. 
In this example, we use the VFNet with ResNext as backbone and Feature Pyramid Network (FPN) as neck.

```{.python}
checkpoint_name = "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco"
num_gpus = 1  # set to -1 to use all GPUs if available
```
You can also use other model by setting `checkpoint_name` to other names. 
Please refer to :ref: `selecting_models` for details about model selection.

As before, we create the MultiModalPredictor with selected checkpoint name and number of GPUs.
We also need to specify the `problem_type` to `"object_detection"`.

```{.python}
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
)
```

### Use a finetuned model
You can also use a previously trained/finetuned predictor to run inference with.
First specify the predictor path, for example:
```{.python}
load_path = "./AutogluonModels/ag-20221104_185342"  # replace this with path to your desired predictor
```
Then load the predictor:
```{.python}
predictor = MultiModalPredictor.load(load_path)
```

## Setting up data

For COCO format data, we need to provide the path for the data split used for inference.

```{.python}
test_path = "./tiny_motorbike_coco/tiny_motorbike/Annotations/test_cocoformat.json"
```

## Running inference
To run inference, perform:

```{.python}
pred = predictor.predict(test_path)
print(pred)
```

The output `pred` is a `pandas` `DataFrame` that has two columns, `image` and `bboxes`, where
- in `image`, each row contains the image path
- in `bboxes`, each row is a list of dictionaries, each one representing a bounding box: 
  - `{"class": <predicted_class_name>, "bbox": [x1, y1, x2, y2], "score": <confidence_score>}`

Here is an example (Note that the real output may contain more rows):
```
                                                image  \
0   ./tiny_motorbike_coco/tiny_motorbike/Annotatio...   
1   ./tiny_motorbike_coco/tiny_motorbike/Annotatio...   
2   ./tiny_motorbike_coco/tiny_motorbike/Annotatio...   
3   ./tiny_motorbike_coco/tiny_motorbike/Annotatio...   
4   ./tiny_motorbike_coco/tiny_motorbike/Annotatio...   
5   ./tiny_motorbike_coco/tiny_motorbike/Annotatio...   
6   ./tiny_motorbike_coco/tiny_motorbike/Annotatio...   
7   ./tiny_motorbike_coco/tiny_motorbike/Annotatio...   
8   ./tiny_motorbike_coco/tiny_motorbike/Annotatio...   
9   ./tiny_motorbike_coco/tiny_motorbike/Annotatio...   
10  ./tiny_motorbike_coco/tiny_motorbike/Annotatio...   

                                               bboxes  
0   [{'class': 'person', 'bbox': [193.2191, 111.00...  
1   [{'class': 'person', 'bbox': [115.89705, 154.7...  
2   [{'class': 'person', 'bbox': [330.596, 1.51525...  
3   [{'class': 'person', 'bbox': [86.17752, 22.335...  
4   [{'class': 'person', 'bbox': [418.78607, 271.4...  
5   [{'class': 'person', 'bbox': [269.31888, 78.00...  
6   [{'class': 'person', 'bbox': [114.42808, 47.03...  
7   [{'class': 'person', 'bbox': [205.9525, 105.30...  
8   [{'class': 'person', 'bbox': [161.37709, 38.26...  
9   [{'class': 'person', 'bbox': [358.3175, 2.5311...  
10  [{'class': 'person', 'bbox': [86.48791, 80.385...   
```

Note that, by default, the `predictor.predict` does not save the detection results into a file.

To run inference and save results, run the following:
```{.python}
pred = predictor.predict(test_path, save_results=True)
```

Currently, we save the `pred`, which is a `pandas` `DataFrame`, into a `.txt` file.
The `.txt` file therefore also has two columns, `image` and `bboxes`, where
- in `image`, each row contains the image path
- in `bboxes`, each row is a list of dictionaries, each one representing a bounding box: 
  - `{"class": <predicted_class_name>, "bbox": [x1, y1, x2, y2], "score": <confidence_score>}`

## Visualizing Results
To run visualizations, ensure that you have `opencv` installed. If you haven't already, install `opencv` by running 
```{.python}
pip install opencv-python
```

To visualize the detection bounding boxes, run the following:
```{.python}
from autogluon.multimodal.utils import from_coco_or_voc, visualize_detection
from matplotlib import pyplot as plt

conf_threshold = 0.4  # Specify a confidence threshold to filter out unwanted boxes
visualization_result_dir = "./tiny_motorbike_coco/tiny_motorbike/visualizations"  # Specify a directory to save visualized images.
df = from_coco_or_voc(test_path)[:10][["image"]]

pred = predictor.predict(df)

visualized = visualize_detection(
    pred=pred,
    detection_classes=predictor.get_predictor_classes(),
    conf_threshold=conf_threshold,
    visualization_result_dir=visualization_result_dir,
)

plt.imshow(visualized[0][:, : ,::-1])  # shows the first image with bounding box
```
Note that we took 10 images to visualize for this example. 
Please consider your storage situation when deciding the number of images to visualize.

The `pred` parameter that `visualize_detection` function takes as input follows the form of a `pandas` `DataFrame`, same as in the `pred_df`. 
Make sure you have the format when visualizing.
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