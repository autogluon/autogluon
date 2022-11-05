# AutoMM Detection - Evaluate Pretrained Faster R-CNN on VOC Format Dataset
:label:`sec_automm_detection_eval_fasterrcnn_voc`

In this section, our goal is to evaluate Faster-RCNN model on VOC2007 dataset in VOC format.
See \[Convert VOC to COCO] for how to quickly convert a VOC format dataset.
In previous section :ref:`sec_automm_detection_eval_fasterrcnn_coco`, we evaluated Faster-RCNN on COCO dataset.
We strongly recommend using COCO format, but AutoMM still have limited support for VOC format for quick proof testing.

To start, let's import MultiModalPredictor:

```python
from autogluon.multimodal import MultiModalPredictor
```

We use the Faster R-CNN with ResNet50 as backbone and Feature Pyramid Network (FPN) as neck.
This is the only model we support that is pretrained on VOC.
It's always recommended to finetune a model pretrained on COCO which is a larger dataset with more complicated task.
To test other model structures on VOC, check \[Fast Finetune on COCO format data] and \[Fast Finetune on VOC format data].

```python
checkpoint_name = "faster_rcnn_r50_fpn_1x_voc0712"
num_gpus = 1  # multi GPU inference is not supported in VOC format
```

As before, we create the MultiModalPredictor with selected checkpoint name and number of GPUs.
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

Here we use VOC2007 for testing: \[Prepare VOC Dataset].
While using VOC format dataset, the input is the root path of the dataset, and contains at least:

```
Annotations  ImageSets  JPEGImages labels.txt
```

Here `labels.txt` shall be added manually to include all the labels in the dataset. 
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

For VOC format data, we always use root_path. And the predictor will automatically select the split.

```python
test_path = "VOCdevkit/VOC2007"
```

To evaluate the pretrained Faster R-CNN model we loaded, run:

```python
result = predictor.evaluate(test_path)
```

Here the test set is selected automatically in `predictor.evaluate`.
And if we `print(result)`, the first value `0.4406` is mAP in COCO standard, and the second one `0.7328` is mAP in VOC standard (or mAP50). For more details about these metrics, see [COCO's evaluation guideline](https://cocodataset.org/#detection-eval).

```
{'map': tensor(0.4406), 'map_50': tensor(0.7328), 'map_75': tensor(0.4658), 'map_small': tensor(0.0959), 'map_medium': tensor(0.3085), 'map_large': tensor(0.5281), 'mar_1': tensor(0.3761), 'mar_10': tensor(0.5297), 'mar_100': tensor(0.5368), 'mar_small': tensor(0.1485), 'mar_medium': tensor(0.4192), 'mar_large': tensor(0.6328), 'map_per_class': tensor(-1.), 'mar_100_per_class': tensor(-1.)}
time usage: 533.67
```

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