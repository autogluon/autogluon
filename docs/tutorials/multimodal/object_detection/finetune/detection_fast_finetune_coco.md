# AutoMM Detection - Fast Finetune on COCO Format Dataset
:label:`sec_automm_detection_fast_ft_coco`

In this section, our goal is to fast finetune a pretrained model on VOC2017 training set, 
and evaluate it in VOC2007 test set. Both training and test sets are in COCO format.
See :ref:`sec_automm_detection_prepare_voc` for how to prepare VOC dataset,
and :ref:`sec_automm_detection_convert_to_coco` for how to convert other datasets to COCO format.

To start, let's import MultiModalPredictor:

```python
from autogluon.multimodal import MultiModalPredictor
```

We select the YOLOv3 with MobileNetV2 as backbone,
and input resolution is 320x320, pretrained on COCO dataset. With this setting, it is fast to finetune or inference,
and easy to deploy.
While using COCO format dataset, the input is the json annotation file of the dataset split.
In this example, `voc07_train.json` and `voc07_test.json` are the annotation files of train and test split of VOC2007 dataset.
And we use all the GPUs (if any):

```python
checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
num_gpus = -1  # use all GPUs

train_path = "./VOCdevkit/VOC2007/Annotations/train_cocoformat.json" 
test_path = "./VOCdevkit/VOC2007/Annotations/test_cocoformat.json"
```

We create the MultiModalPredictor with selected checkpoint name and number of GPUs.
We need to specify the problem_type to `"object_detection"`,
and also provide a `sample_data_path` for the predictor to infer the catgories of the dataset.
Here we provide the `train_path`, and it also works using any other split of this dataset.

```python
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
    sample_data_path=train_path,
)
```

If no data sample is available at this point, you can also create the MultiModalPredictor by manually input the classes:

```python
voc_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
    classes=voc_classes,
)
```

We set the learning rate to be `1e-4`.
Note that we use a two-stage learning rate option during finetuning by default,
and the model head will have 100x learning rate.
Using a two-stage learning rate with high learning rate only on head layers makes
the model converge faster during finetuning. It usually gives better performance as well,
especially on small datasets with hundreds or thousands of images.
We also set the epoch to be 5 for fast finetuning and batch_size to be 32.
We also compute the time of the fit process here for better understanding the speed.
```python
import time
start = time.time()
predictor.fit(
    train_path,
    hyperparameters={
        "optimization.learning_rate": 1e-4, # we use two stage and detection head has 100x lr
        "optimization.max_epochs": 5,
        "env.per_gpu_batch_size": 32,  # decrease it when model is large
    },
)
end = time.time()
```

We run it on a g5dn.12xlarge EC2 machine on AWS,
and part of the command outputs are shown below:

```
Epoch 0:  98%|██████████████████████████████████████████████████████████████████████████████████████████▏ | 50/51 [00:15<00:00,  3.19it/s, loss=766, v_num=Epoch 0, global step 40: 'val_direct_loss' reached 555.37537 (best 555.37537), saving model to '/media/code/autogluon/examples/automm/object_detection/AutogluonModels/ag-20221104_185342/epoch=0-step=40.ckpt' as top 1
Epoch 1:  49%|█████████████████████████████████████████████                                               | 25/51 [00:08<00:08,  3.01it/s, loss=588, v_num=Epoch 1, global step 61: 'val_direct_loss' reached 499.56232 (best 499.56232), saving model to '/media/code/autogluon/examples/automm/object_detection/AutogluonModels/ag-20221104_185342/epoch=1-step=61.ckpt' as top 1
Epoch 1:  98%|██████████████████████████████████████████████████████████████████████████████████████████▏ | 50/51 [00:15<00:00,  3.17it/s, loss=554, v_num=Epoch 1, global step 81: 'val_direct_loss' reached 481.33121 (best 481.33121), saving model to '/media/code/autogluon/examples/automm/object_detection/AutogluonModels/ag-20221104_185342/epoch=1-step=81.ckpt' as top 1
Epoch 2:  49%|█████████████████████████████████████████████                                               | 25/51 [00:08<00:08,  2.99it/s, loss=539, v_num=Epoch 2, global step 102: 'val_direct_loss' reached 460.25449 (best 460.25449), saving model to '/media/code/autogluon/examples/automm/object_detection/AutogluonModels/ag-20221104_185342/epoch=2-step=102.ckpt' as top 1
Epoch 2:  98%|██████████████████████████████████████████████████████████████████████████████████████████▏ | 50/51 [00:15<00:00,  3.15it/s, loss=539, v_num=Epoch 2, global step 122: 'val_direct_loss' was not in top 1                                                                                                 
Epoch 3:  49%|█████████████████████████████████████████████                                               | 25/51 [00:08<00:08,  2.96it/s, loss=533, v_num=Epoch 3, global step 143: 'val_direct_loss' was not in top 1                                                                                                 
Epoch 3:  88%|█████████████████████████████████████████████████████████████████████████████████▏          | 45/51 [00:14<00:01,  3.17it/s, loss=508, v_num=]
```

Notice that at the end of each progress bar, if the checkpoint at current stage is saved,
it prints the model's save path.
In this example, it's `/media/code/autogluon/examples/automm/object_detection/AutogluonModels/ag-20221104_185342`.
You can also specify the `save_path` like below while creating the MultiModalPredictor.

```
predictor = MultiModalPredictor(
    save_path="./this_is_a_save_path",
    ...
)
```

Print out the time and we can see that it only takes 100.42 seconds!

```python
print("This finetuning takes %.2f seconds." % (end - start))
```

```
This finetuning takes 100.42 seconds.
```

To evaluate the model we just trained, run:

```python
predictor.evaluate(test_path)
```

And the evaluation results are shown in command line output. 
The first value `0.375` is mAP in COCO standard, and the second one `0.755` is mAP in VOC standard (or mAP50). 
For more details about these metrics, see [COCO's evaluation guideline](https://cocodataset.org/#detection-eval).

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.755
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.111
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.230
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.258
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.556
```

Under this fast finetune setting, we reached `mAP50 = 0.755` on VOC with 100 seconds!
For how to finetune with higher performance,
see :ref:`sec_automm_detection_high_ft_coco`, where we finetuned a VFNet model with 
5 hours and reached `mAP50 = 0.932` on VOC.

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
