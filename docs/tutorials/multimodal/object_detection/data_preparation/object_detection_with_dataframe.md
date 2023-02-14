# AutoMM Detection - Object detection data formats
:label:`sec_automm_detection_df_to_coco_and_back`

In this section, we introduce the two major data formats that AutoMM Detection supports, which are COCO format and DataFrame format.


## COCO Format
See section :ref:`sec_automm_detection_convert_to_coco` for a detailed introduction on the COCO dataset format. 
Essentially you will need a `.json` file that holds data information for your dataset. 
For example, you could prepare your data in the following format:

```
data = {
    # list of dictionaries containing all the category information
    "categories": [
        {"supercategory": "none", "id": 1, "name": "person"},
        {"supercategory": "none", "id": 2, "name": "bicycle"},
        {"supercategory": "none", "id": 3, "name": "car"},
        {"supercategory": "none", "id": 4, "name": "motorcycle"},
        # ...
    ],

    # list of dictionaries containing image info
    "images": [
        {
            "file_name": "<imagename0>.<ext>",
            "height": 427,
            "width": 640,
            "id": 1
        },
        {
            "file_name": "<imagename2>.<ext>",
            "height": 427,
            "width": 640,
            "id": 2
        },
        # ...
    ],
    # list of dictionaries containing bounding box annotation info
    "annotations": [
        {
            'area': 33453,  # area of the bounding box
            'iscrowd': 0,  # if the bounding box contains multiple objects, usually this is 0 since we are dealing with single box -> single object 
            'bbox': [181, 133, 177, 189],  # the [x, y, width, height] format annotation of bounding box
            'category_id': 8,  # the "id" field of the corresponding category, not the "name" field
            'ignore': 0,  # set to 1 to ignore this annotation
            'segmentation': [],  # always empty since this tutorial is not for segmentation
            'image_id': 1617,  # the "id" field of the corresponding image
            'id': 1  # the "id" of this particular annotation
        },
        {
            'area': 25740, 
            'iscrowd': 0,
            'bbox': [192, 100, 156, 165],
            'category_id': 9,
            'ignore': 0,
            'segmentation': [],
            'image_id': 1617,
            'id': 2
        },
        # ...
    ],
    
    "type": "instances"
}
```


## `pd.DataFrame` Format
The AutoMM detection also supports the `pd.DataFrame` format. Your `pd.DataFrame` should contain 3 columns. 

- `image`: the path to the image file
- `rois`: a list of arrays containing bounding box annotation `[x1, y1, x2, y2, class_label]`
- `label`: a copy column of `rois`

An example can be seen below:
```
                                               image  \
0  /home/ubuntu/autogluon-dev/docs/tutorials/mult...   
1  /home/ubuntu/autogluon-dev/docs/tutorials/mult...   
2  /home/ubuntu/autogluon-dev/docs/tutorials/mult...   
3  /home/ubuntu/autogluon-dev/docs/tutorials/mult...   
4  /home/ubuntu/autogluon-dev/docs/tutorials/mult...   

                                                rois  \
0  [[352.0, 138.0, 374.0, 373.0, 7], [105.0, 1.0,...   
1  [[40.0, 71.0, 331.0, 332.0, 7], [33.0, 42.0, 3...   
2  [[52.0, 22.0, 306.0, 326.0, 8], [26.0, 108.0, ...   
3  [[114.0, 154.0, 367.0, 346.0, 7], [292.0, 49.0...   
4  [[279.0, 225.0, 374.0, 338.0, 3], [245.0, 230....   

                                               label  
0  [[352.0, 138.0, 374.0, 373.0, 7], [105.0, 1.0,...  
1  [[40.0, 71.0, 331.0, 332.0, 7], [33.0, 42.0, 3...  
2  [[52.0, 22.0, 306.0, 326.0, 8], [26.0, 108.0, ...  
3  [[114.0, 154.0, 367.0, 346.0, 7], [292.0, 49.0...  
4  [[279.0, 225.0, 374.0, 338.0, 3], [245.0, 230....  
```

## Using the data formats to train and evaluate models

### Download data
We have the sample dataset ready in the cloud. Let's download it:
```python
import os
import time

from autogluon.core.utils.loaders import load_zip

zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
download_dir = "./tiny_motorbike_coco"

load_zip.unzip(zip_file, unzip_dir=download_dir)
data_dir = os.path.join(download_dir, "tiny_motorbike")
train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
```

We provide useful util functions to convert from COCO format to `pd.DataFrame` format and vice versa.

### From COCO format to `pd.DataFrame`
Now we first introduce converting from COCO to `pd.DataFrame`

```python
from autogluon.multimodal.utils.object_detection import from_coco
train_df = from_coco(train_path)
print(train_df)
```

### From `pd.DataFrame` to COCO format
```python
from autogluon.multimodal.utils.object_detection import object_detection_df_to_coco
train_coco = object_detection_df_to_coco(train_df)
print(train_coco)
```
You can save the `train_coco`, which is a dictionary, to a `.json` file by specifying the `save_path` when calling `object_detection_df_to_coco`.
```python
train_coco = object_detection_df_to_coco(train_df, save_path="./df_converted_to_coco.json")
```

The next time when loading from the `.json` file by calling `from_coco`, make sure to supply the right `root` such that `<root>/<file_name>` is a valid image path.
(Note: `file_name` is under the `"images"` subfield in `data` defined at the beginning of this tutorial.) For example:
```python
train_df_from_saved_coco = from_coco("./df_converted_to_coco.json", root="./")
```

### Training with `pd.DataFrame` format
To start, let's import MultiModalPredictor:

```python .input
from autogluon.multimodal import MultiModalPredictor
```

Make sure `mmcv-full` and `mmdet` are installed:
```python .input
!mim install mmcv-full
!pip install mmdet
```

Again, we follow the model setup as in :ref:`sec_automm_detection_quick_start_coco` 
```python
checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
num_gpus = -1  # use all GPUs
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-df_train_temp_save"
predictor_df = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
    sample_data_path=train_df,  # we specify train_df here as the sample_data_path in order to get the num_classes
    path=model_path,
)

predictor_df.fit(
    train_df,
    hyperparameters={
        "optimization.learning_rate": 2e-4, # we use two stage and detection head has 100x lr
        "optimization.max_epochs": 30,
        "env.per_gpu_batch_size": 32,  # decrease it when model is large
    },
)
```

### Evaluation with `pd.DataFrame` format
We follow the evaluation setup as in :ref:`sec_automm_detection_quick_start_coco`. We encourage you to check it out for further details.   

To evaluate the model with `pd.DataFrame` format, run following code.

```python
test_df = from_coco(test_path)
predictor_df.evaluate(test_df)
```

### Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

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
