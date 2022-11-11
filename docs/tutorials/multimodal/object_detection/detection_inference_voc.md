# AutoMM Detection - Inference with Pretrained Faster R-CNN on VOC Format Dataset
:label:`sec_automm_detection_infer_voc`

In this section, we show an example to run inference on VOC2007 dataset in VOC Format. 
Different from running evaluation, the purpose is to get detection results for potential down-stream tasks. 
For more details about the VOC format, please see \[Convert VOC to COCO].
However, we strongly recommend using the COCO format, through we provide limited support for VOC format.

[//]: # (In this section, our goal is to evaluate Faster-RCNN model on VOC2007 dataset in VOC format.)

[//]: # (See \[Convert VOC to COCO] for how to quickly convert a VOC format dataset.)

[//]: # (In previous section :ref:`sec_automm_detection_eval_fasterrcnn_coco`, we evaluated Faster-RCNN on COCO dataset.)

[//]: # (We strongly recommend using COCO format, but AutoMM still have limited support for VOC format for quick proof testing.)

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
Please refer to :ref: `selecting_models` for details about model selection. 

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
load_path = "./AutogluonModels/ag-20221104_185342"
```
Then load the predictor:
```python
predictor = MultiModalPredictor.load(load_path)
```

## Setting up data

Here we use VOC2007 for testing: \[Prepare VOC Dataset].
While using VOC format dataset, the input is the root path of the dataset, and contains at least:

```
Annotations  ImageSets  JPEGImages labels.txt
```

[//]: # (Here `labels.txt` shall be added manually to include all the labels in the dataset.)
Here `labels.txt` includes all the labels in the dataset. If it does not exist, the code will generate it by scanning 
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

For VOC format data, we always use root_path. And the predictor will automatically select the test split.

```python
test_path = "VOCdevkit/VOC2007"
```

## Running inference
To run inference, run:

```python
pred = predictor.predict(test_path)
```

To run inference and save results, run the following:
```python
pred = predictor.predict(test_path, save_results=True)
```
Currently, we support saving results into a `.txt` file as a pandas `DataFrame`.

Again, the test set is selected automatically within `predictor.predict` function.

The returned value `pred` is a `list` and has the following dimensions:
```python
[num_images, num_total_classes, num_detections_per_class, 5]
```

where 
- `num_images` is the number of images used to run inference on. 
- `num_total_classes` is the total number of classes. In this example, `num_total_classes = 20` for VOC dataset
- `num_detections_per_class` is the number of detections under each class. Note that this number can vary across different classes.
- The last dimension contains the bounding box information, which follows `[x1, y1, x2, y2, score]` format. `x1, y1` are the top left corner of the bounding box, and `x2, y2` are the bottom right corner. `score` is the confidence score of the prediction

example code to examine bounding box information:

```python
detection_classes = predictor.get_predictor_classes()
idx2classname = {idx: classname for (idx, classname) in enumerate(detection_classes)}
for i, image_pred in enumerate(pred):
    print("result for image {}".format(i))
    for j, per_cls_bboxes in enumerate(image_pred):
        classname = idx2classname[j]
        for bbox in per_cls_bboxes:
            ## bbox = [x1, y1, x2, y2, conf_score]
            print("bbox: {}, class: {}, score: {}".format(bbox[:4], classname, bbox[4]))
```

## Visualizing Results
To visualize the detection bounding boxes, run the following:
```python
from autogluon.multimodal.utils import from_coco_or_voc, visualize_detection

conf_threshold = 0.4  # Specify a confidence threshold to filter out unwanted boxes
visualization_result_dir = "VOCdevkit/VOC2007/visualizations"  # Specify a directory to save visualized images.

df = from_coco_or_voc(test_path)[:10][["image"]]  # we took 10 images for this example.

pred = predictor.predict(df)

visualize_detection(
    pred=pred,
    data=df,
    detection_classes=predictor.get_predictor_classes(),
    conf_threshold=conf_threshold,
    visualization_result_dir=visualization_result_dir,
)
```
Note that we took 10 images to visualize for this example. 
Please consider your storage situation when deciding the number of images to visualize.  

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