# AutoMM Detection - Inference with Pretrained Faster R-CNN on VOC Dataset
:label:`sec_automm_detection_infer_voc`

In this section, we show an example to run inference on VOC2007 dataset that is in VOC Format. 
Different from running evaluation, the purpose is to get detection results for potential down-stream tasks. 
For more details about the data format, please see :ref:`sec_automm_detection_convert_to_coco`.
However, we strongly recommend using the COCO format, through we provide limited support for VOC format.
For information about converting VOC format to COCO format, please see :ref:`sec_automm_detection_voc_to_coco`
We use a Faster R-CNN model model pretrained on VOC2007 and VOC2012 dataset

AutoMM detection requires `mmcv-full` and `mmdet` packages. Please make sure `mmcv-full` and `mmdet` are installed:
```python
!mim install mmcv-full
!pip install mmdet
```

## Prepare data
For running this tutorial, you should have VOC dataset prepared. 
If you haven't already, head over to :ref:`sec_automm_detection_prepare_coco17` to prepare your VOC data. 
This tutorial assumes that you have saved VOC data under the folder `~/data/`, i.e. it should appear at `~/data/VOCdevkit`.

## Creating the `MultiModalPredictor`
To start, import MultiModalPredictor:

```python
from autogluon.multimodal import MultiModalPredictor
```

### Use a pretrained model
You can download a pretrained model and construct a predictor with it.
In this tutorial, we use the Faster R-CNN with ResNet50 as backbone and Feature Pyramid Network (FPN) as neck.

This is the only model we support that is pretrained on VOC.
It's always recommended to finetune a model pretrained on COCO which is a larger dataset with more complicated task.
To test other model structures on VOC, check \[Fast Finetune on COCO format data] and \[Fast Finetune on VOC format data].

```python
checkpoint_name = "faster_rcnn_r50_fpn_1x_voc0712"
num_gpus = 1  # multi GPU inference is not supported in VOC format
```
You can also use other model by setting `checkpoint_name` to other model configs.  

As before, we create the MultiModalPredictor with selected checkpoint name and number of GPUs.
We also need to specify the `problem_type` to `"object_detection"`.

```python
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
```python
load_path = "./AutogluonModels/ag-20221104_185342"  # replace this with path to your desired predictor
```
Then load the predictor:
```python
predictor = MultiModalPredictor.load(load_path)
```

## Setting up data

Here we use VOC2007 for testing. See :ref:`sec_automm_detection_prepare_voc` for VOC data preparation.
While using VOC format dataset, the input is the root path of the dataset, and contains at least:

```
Annotations  ImageSets  JPEGImages labels.txt
```

[//]: # (Here `labels.txt` shall be added manually to include all the labels in the dataset.)
Here `labels.txt` includes all the labels in the dataset. If not existed, it will be automatically generate it by scanning 
through all annotation files.
In this example, the content of `labels.txt` is shown as below:

```
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
```

For VOC format data, we always use root_path. And the predictor will automatically select the `test` split.

```python
test_path = "~/data/VOCdevkit/VOC2007"
```

## Running inference
To run inference, perform:

```python
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
0     /home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages...   
1     /home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages...   
2     /home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages...   
3     /home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages...   
4     /home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages...   
...                                                 ...   
4947  /home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages...   
4948  /home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages...   
4949  /home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages...   
4950  /home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages...   
4951  /home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages...   

                                                 bboxes  
0     [{'class': 'dog', 'bbox': [52.044403, 229.2540...  
1     [{'class': 'aeroplane', 'bbox': [79.59477, 53....  
2                                                    []  
3     [{'class': 'car', 'bbox': [363.55017, 326.7566...  
4     [{'class': 'chair', 'bbox': [133.64464, 193.56...  
...                                                 ...  
4947  [{'class': 'cat', 'bbox': [105.409004, 10.5548...  
4948  [{'class': 'diningtable', 'bbox': [18.526844, ...  
4949  [{'class': 'bicycle', 'bbox': [101.61986, 98.3...  
4950  [{'class': 'cat', 'bbox': [99.85065, 282.5224,...  
4951  [{'class': 'aeroplane', 'bbox': [15.229627, 0....  

[4952 rows x 2 columns]
```

Note that, by default, the `predictor.predict` does not save the detection results into a file.

To run inference and save results, run the following:
```python
pred = predictor.predict(test_path, save_results=True)
```

Currently, we save the `pred`, which is a `pandas` `DataFrame`, into a `.txt` file.
The `.txt` file therefore also has two columns, `image` and `bboxes`, where
- in `image`, each row contains the image path
- in `bboxes`, each row is a list of dictionaries, each one representing a bounding box: 
  - `{"class": <predicted_class_name>, "bbox": [x1, y1, x2, y2], "score": <confidence_score>}`

## Visualizing Results
To run visualizations, ensure that you have `opencv` installed. If you haven't already, install `opencv` by running 
```python
!pip install opencv-python
```

To visualize the detection bounding boxes, run the following:
```python
from autogluon.multimodal.utils import visualize_detection
conf_threshold = 0.4  # Specify a confidence threshold to filter out unwanted boxes
visualization_result_dir = "~/data/VOCdevkit/VOC2007/visualizations"  # Specify a directory to save visualized images.

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
@article{Ren_2017,
   title={Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
   year={2017},
   month={Jun},
}
```