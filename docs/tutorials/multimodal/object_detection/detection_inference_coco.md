# AutoMM Detection - Inference with Pretrained Faster R-CNN on VOC Format Dataset
:label:`sec_automm_detection_infer_coco`

In this section, we show an example to run inference COCO dataset in COCO Format. 
Different from running evaluation, the purpose is to get detection results for potential down-stream tasks.

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
In this example, we use the VFNet with ResNext as backbone and Feature Pyramid Network (FPN) as neck.

```python
checkpoint_name = "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco"
num_gpus = -1  # using all GPUs if available
```
You can also use other model by setting `checkpoint_name` to other names. 
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

For COCO format data, we need to provide the path for the data split used for inference.

```python
test_path = "coco17/annotations/instances_val2017.json"
```

## Running inference
To run inference, perform:

```python
pred = predictor.predict(test_path)
```

To run inference and save results, run the following:
```python
pred = predictor.predict(test_path, save_results=True)
```
Currently, we support saving results into a `.txt` file as a pandas `DataFrame`.

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
visualization_result_dir = "coco17/visualizations"  # Specify a directory to save visualized images.
df = from_coco_or_voc(test_path)[:10][["image"]]

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