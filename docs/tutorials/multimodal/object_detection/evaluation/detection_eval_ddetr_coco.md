# AutoMM Detection - Evaluate Pretrained Deformable DETR on COCO Format Dataset
:label:`sec_automm_detection_eval_ddetr_coco`

In this section, our goal is to evaluate Deformable DETR model on COCO17 dataset in COCO format. 
Previously we introduced two classic models: :ref:`sec_automm_detection_eval_yolov3_coco` and :ref:`sec_automm_detection_eval_fasterrcnn_coco`.
Recent years Transformer models become more and more popular in Computer Vision, and Deformable DEtection TRansformer (Deformable DETR) reached the SOTA performance in detection task.
In terms of speed, it's slower than YOLOv3 and Faster-RCNN, but it also has higher performance.

To start, let's import MultiModalPredictor:

```python
from autogluon.multimodal import MultiModalPredictor
```

We select the two-stage Deformable DETR with ResNet50 as backbone with bounding box finetune,
this model has **15.7 frames per second (FPS)** on single A10e GPU with `batch_size=1`.
And we use all the GPUs (if any):

```python
checkpoint_name = "deformable_detr_twostage_refine_r50_16x2_50e_coco"
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
See other tutorials for :ref:`sec_automm_detection_prepare_coco17`.
While using COCO dataset, the input is the json annotation file of the dataset split.
In this example, `instances_val2017.json` is the annotation file of validation split of COCO17 dataset.

```python
test_path = "coco17/annotations/instances_val2017.json"
```

To evaluate the pretrained Deformable DETR model we loaded, run:

```python
predictor.evaluate(test_path)
```

And the evaluation results are shown in command line output. The first value `0.463` is mAP in COCO standard, and the second one `0.659` is mAP in VOC standard (or mAP50). For more details about these metrics, see [COCO's evaluation guideline](https://cocodataset.org/#detection-eval).

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.463
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.659
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.500
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.298
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.493
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.607
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.603
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.652
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.451
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.692
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.830
time usage: 389.92
```

Deformable DETR has best performance but takes more time and (GPU memory) space. 
If there is a restriction in time or space, 
see :ref:`sec_automm_detection_eval_fasterrcnn_coco` or :ref:`sec_automm_detection_eval_yolov3_coco`.
You can also see other tutorials for :ref:`sec_automm_detection_high_ft_coco` or :ref:`sec_automm_detection_infer_coco`.

### Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

### Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.

### Citation
```
@inproceedings{
zhu2021deformable,
title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
author={Xizhou Zhu and Weijie Su and Lewei Lu and Bin Li and Xiaogang Wang and Jifeng Dai},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=gZ9hCDWe6ke}
}
```