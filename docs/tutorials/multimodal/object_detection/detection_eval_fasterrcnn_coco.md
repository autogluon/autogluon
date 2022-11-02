# AutoMM Detection - Evaluate Pretrained Faster R-CNN on COCO Format Dataset
:label:`sec_automm_detection_eval_fasterrcnn_coco`

In this section, our goal is to evaluate Faster-RCNN model on VOC2007 dataset in COCO format.
In previous section :ref:`sec_automm_detection_eval_yolov3_coco`, we evaluated YOLOv3 which is small and fast. But if higher performance is required, we need a bigger model.
And Faster R-CNN has a balance in speed and performance trade-offs.

To start, import MultiModalPredictor:

```python
from autogluon.multimodal import MultiModalPredictor
```

We select the Faster R-CNN with ResNet50 as backbone and Feature Pyramid Network (FPN)  as neck,
for other Faster R-CNN models see [MMDetection's Faster R-CNN models](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn). 
And we still use all the GPUs (if any):

```python
checkpoint_name = "faster_rcnn_r50_fpn_2x_coco"
num_gpus = -1  # use all GPUs
```

As before, we create the MultiModalPredictor with selected checkpoint name and number of GPUs.
We also need to specify the problem_type is `"object_detection"`.

```python
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
)
```

Here we use COCO17 to test. 
See other tutorials for \[Prepare Common Public Dataset], \[Convert VOC Format Dataset to COCO Format], and \[Create Custom Dataset].
While using COCO dataset, the input is the json annotation file of the dataset split.
In this example, `instances_val2017.json` is the annotation file of validation split of COCO17 dataset.

```python
test_path = "coco17/annotations/instances_val2017.json"
```

To evaluate the pretrained Faster R-CNN model we loaded, run:

```python
predictor.evaluate(test_path)
```

And the evaluation results is shown in command line output. The first value `0.385` is mAP in COCO standard, and the second one `0.591` is mAP in VOC standard (or mAP50). For more details about these metrics, see [COCO's evaluation guideline](https://cocodataset.org/#detection-eval).

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.591
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.326
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.558
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.667
time usage: 257.45
```

Faster R-CNN balances speed and performance. 
But in case that faster speed or higher performance is required, 
see :ref:`sec_automm_detection_eval_yolov3_coco` for faster speed,
or :ref:`sec_automm_detection_eval_ddetr_coco` for higher performance.
You can also see other tutorials for \[Fast Finetune on COCO format data] or \[Inference on COCO format data (with Visualization)].

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