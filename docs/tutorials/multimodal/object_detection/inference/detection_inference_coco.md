# AutoMM Detection - Inference with Pretrained VFNet on COCO Dataset
:label:`sec_automm_detection_infer_coco`

In this section, we show an example to run inference COCO dataset in COCO Format. 
Different from running evaluation, the purpose is to get detection results for potential down-stream tasks.
The model we use is the VFNet pretrained on COCO dataset.

AutoMM detection requires `mmcv-full` and `mmdet` packages. Please make sure `mmcv-full` and `mmdet` are installed:
```{.python}
!mim install mmcv-full
!pip install mmdet
```

## Prepare data
For running this tutorial, you should have COCO dataset prepared.
If you haven't already, head over to :ref:`sec_automm_detection_prepare_coco17` to learn how to get COCO dataset.
This tutorial assumes that COCO dataset is under `~/data`, i.e. it is located in `~/data/coco17`

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
test_path = "~/data/coco17/annotations/instances_val2017.json"
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

Here is an example:
```
                                                  image  \
0     /home/ubuntu/data/coco17/annotations/../val201...   
1     /home/ubuntu/data/coco17/annotations/../val201...   
2     /home/ubuntu/data/coco17/annotations/../val201...   
3     /home/ubuntu/data/coco17/annotations/../val201...   
4     /home/ubuntu/data/coco17/annotations/../val201...   
...                                                 ...   
4947  /home/ubuntu/data/coco17/annotations/../val201...   
4948  /home/ubuntu/data/coco17/annotations/../val201...   
4949  /home/ubuntu/data/coco17/annotations/../val201...   
4950  /home/ubuntu/data/coco17/annotations/../val201...   
4951  /home/ubuntu/data/coco17/annotations/../val201...   

                                                 bboxes  
0     [{'class': 'person', 'bbox': [410.91803, 156.5...  
1     [{'class': 'bird', 'bbox': [0.11992988, 66.977...  
2     [{'class': 'bottle', 'bbox': [96.73349, 192.20...  
3     [{'class': 'person', 'bbox': [78.57298, 272.01...  
4     [{'class': 'person', 'bbox': [0.6821054, 53.59...  
...                                                 ...  
4947  [{'class': 'person', 'bbox': [411.93835, 54.90...  
4948  [{'class': 'person', 'bbox': [257.81232, 94.21...  
4949  [{'class': 'sports ball', 'bbox': [370.05905, ...  
4950  [{'class': 'person', 'bbox': [257.79666, 86.02...  
4951  [{'class': 'bowl', 'bbox': [201.58173, 183.721...  

[4952 rows x 2 columns]
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
!pip install opencv-python
```

To visualize the detection bounding boxes, run the following:
```{.python}
from autogluon.multimodal.utils import visualize_detection

conf_threshold = 0.4  # Specify a confidence threshold to filter out unwanted boxes
visualization_result_dir = "~/data/coco17/visualizations"  # Specify a directory to save visualized images.

visualized = visualize_detection(
    pred=pred.iloc[:10],  # we took the first 10 images for this example
    detection_classes=predictor.get_predictor_classes(),
    conf_threshold=conf_threshold,
    visualization_result_dir=visualization_result_dir,
)

from PIL import Image
from IPython.display import display
img = Image.fromarray(visualized[0][:, :, ::-1], 'RGB')
display(img)
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