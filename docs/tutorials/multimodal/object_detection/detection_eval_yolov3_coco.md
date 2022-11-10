# AutoMM Detection - Evaluate Pretrained YOLOv3 on COCO Format Dataset
:label:`sec_automm_detection_eval_yolov3_coco`

In this section, our goal is to evaluate YOLOv3 model on COCO17 dataset in COCO format. We start with yolov3 because it's extremely fast and accurate, and is a good choice to deploy with strict time and computational restrictions.

To start, let's import MultiModalPredictor:

```python
from autogluon.multimodal import MultiModalPredictor
```

We select the YOLOv3 with MobileNetV2 as backbone, 
this model reached **85.0 Frames Per Second (FPS)** on single A10e GPU with `batch_size=1`.
For details about model selection, see :ref:`sec_automm_detection_select_models`.
And we use all the GPUs (if any):

```python
checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
num_gpus = -1  # use all GPUs
```

We create the MultiModalPredictor with selected checkpoint name and number of GPUs.
We also need to specify the problem_type to `"object_detection"`.

```python
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
)
```

Here we use COCO17 for testing. 
See other tutorials for \[Prepare COCO2017], \[Convert VOC Format Dataset to COCO Format], and \[Create Custom Dataset].
While using COCO dataset, the input is the json annotation file of the dataset split.
In this example, `instances_val2017.json` is the annotation file of validation split of COCO17 dataset.

```python
test_path = "coco17/annotations/instances_val2017.json"
```

To evaluate the pretrained YOLOv3 model we loaded, run:

```python
predictor.evaluate(test_path)
```

And the evaluation results are shown in command line output. The first value `0.223` is mAP in COCO standard, and the second one `0.420` is mAP in VOC standard (or mAP50). For more details about these metrics, see [COCO's evaluation guideline](https://cocodataset.org/#detection-eval).

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.223
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.060
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.215
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.137
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.385
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529
time usage: 81.76
``

YOLOv3 is small and fast. For larger model with higher performance, see :ref:`sec_automm_detection_eval_fasterrcnn_coco` or :ref:`sec_automm_detection_eval_ddetr_coco`.
You can also see other tutorials for \[Fast Finetune on COCO format data] or \[Inference on COCO format data (with Visualization)].

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
