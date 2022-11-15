# AutoMM Detection - High Performance Finetune on COCO Format Dataset
:label:`sec_automm_detection_high_performance_finetune_coco`

In this section, our goal is to finetune a high performance model on VOC2017 training set, 
and evaluate it in VOC2007 test set. Both training and test sets are in COCO format.
See :ref:`sec_automm_detection_prepare_voc` for how to prepare VOC dataset,
and :ref:`sec_automm_detection_convert_to_coco` for how to convert other datasets to COCO format.

To start, let's import MultiModalPredictor:

```python
from autogluon.multimodal import MultiModalPredictor
```

We select the VFNet with ResNeXt-101 as backbone, Feature Pyramid Network (FPN) as neck,
and input resolution is 640x640, pretrained on COCO dataset.
*(The neck of the object detector refers to the additional layers existing between the backbone and the head. 
Their role is to collect feature maps from different stages.)*
With this setting, it sacrifices training and inference time,
and also requires much more GPU memory,
but the performance is high. 
For more model choices, see :label:`sec_automm_detection_selecting_models`.

We use `val_metric = map`, i.e., mean average precision in COCO standard as our validation metric.
In previous section :ref:`sec_automm_detection_fast_finetune_coco`,
we did not specify the validation metric and by default the validation loss is used as validation metric.
Using validation loss is much faster but using mean average precision gives the best performance.

While using COCO format dataset, the input is the json annotation file of the dataset split.
In this example, `voc07_train.json` and `voc07_test.json` 
are the annotation files of train and test split of VOC2007 dataset.
And we use all the GPUs (if any):

```python
checkpoint_name = "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco"
num_gpus = -1  # use all GPUs
val_metric = "map"

train_path = "./VOCdevkit/VOC2007/Annotations/train_cocoformat.json" 
test_path = "./VOCdevkit/VOC2007/Annotations/test_cocoformat.json"
```

We create the MultiModalPredictor with selected checkpoint name, val_metric, and number of GPUs.
We need to specify the problem_type to `"object_detection"`,
and also provide a `sample_data_path` for the predictor to infer the catgories of the dataset.
Here we provide the `train_path`, and it also works using any other split of this dataset.

```python
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
        "optimization.val_metric": val_metric,
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
        "optimization.val_metric": val_metric,
    },
    problem_type="object_detection",
    classes=voc_classes,
)
```

We set the learning rate to be `1e-5` and epoch to be 20 for fast finetuning.
Note that we use a two-stage learning rate option during finetuning by default,
and the model head will have 100x learning rate.
Using a two-stage learning rate with high learning rate only on head layers makes
the model converge faster during finetuning. It usually gives better performance as well,
especially on small datasets with hundreds or thousands of images.
We also set the batch_size to be 2, because this model is too huge to run with larger batch size.
For more information about how to tune those hyperparameters,
see :ref:`sec_automm_detection_tune_hyperparameters`.
We also compute the time of the fit process here for better understanding the speed.
```python
import time
start = time.time()
predictor.fit(
    train_path,
    hyperparameters={
        "optimization.learning_rate": 1e-5, # we use two stage and detection head has 100x lr
        "optimization.max_epochs": 20,
        "env.per_gpu_batch_size": 1,  # decrease it when model is large
    },
)
end = time.time()
```

We run it on a g5dn.12xlarge EC2 machine on AWS,
and part of the command outputs are shown below:

```
Epoch 0:  50%|███████████████████████████████████████████▌                                           | 394/788 [07:42<07:42,  1.17s/it, loss=1.52, v_num=Epoch 0, global step 20: 'val_map' reached 0.61814 (best 0.61814), saving model to '/media/code/autogluon/examples/automm/object_detection/AutogluonModels/ag-20221104_051558/epoch=0-step=20.ckpt' as top 1                                                                                                                     
Epoch 0: 100%|██████████████████████████████████████████████████████████████████████████████████████| 788/788 [15:29<00:00,  1.18s/it, loss=0.982, v_num=Epoch 0, global step 41: 'val_map' reached 0.68742 (best 0.68742), saving model to '/media/code/autogluon/examples/automm/object_detection/AutogluonModels/ag-20221104_051558/epoch=0-step=41.ckpt' as top 1                                                                                                                     
Epoch 1:  50%|████████████████████████████████████████████                                            | 394/788 [07:54<07:54,  1.20s/it, loss=0.879, v_numEpoch 1, global step 61: 'val_map' reached 0.70111 (best 0.70111), saving model to '/media/code/autogluon/examples/automm/object_detection/AutogluonModels/ag-20221104_051558/epoch=1-step=61.ckpt' as top 1                                                                                                                    
Epoch 1: 100%|████████████████████████████████████████████████████████████████████████████████████████| 788/788 [15:49<00:00,  1.21s/it, loss=0.759, v_num=Epoch 1, global step 82: 'val_map' reached 0.70580 (best 0.70580), saving model to '/media/code/autogluon/examples/automm/object_detection/AutogluonModels/ag-20221104_051558/epoch=1-step=82.ckpt' as top 1                                                                                                                   
Epoch 2:  50%|████████████████████████████████████████████▌                                            | 394/788 [07:47<07:47,  1.19s/it, loss=1.11, v_num=Epoch 2, global step 102: 'val_map' was not in top 1                                                                                                             
Epoch 2: 100%|████████████████████████████████████████████████████████████████████████████████████████| 788/788 [15:29<00:00,  1.18s/it, loss=0.712, v_num=Epoch 2, global step 123: 'val_map' reached 0.71277 (best 0.71277), saving model to '/media/code/autogluon/examples/automm/object_detection/AutogluonModels/ag-20221104_051558/epoch=2-step=123.ckpt' as top 1                                                                                                                 
Epoch 3:  50%|████████████████████████████████████████████▌                                            | 394/788 [07:38<07:38,  1.16s/it, loss=1.07, v_num=Epoch 3, global step 143: 'val_map' was not in top 1                                                                                                             
```

Notice that at the end of each progress bar, if the checkpoint at current stage is saved,
it prints the model's save path.
In this example, it's `/media/code/autogluon/examples/automm/object_detection/AutogluonModels/ag-20221104_051558`.
You can also specify the `save_path` like below while creating the MultiModalPredictor.
For more information about save and load the model,
see :ref:`sec_automm_detection_save_and_load`. 

```
predictor = MultiModalPredictor(
    save_path="./this_is_a_save_path",
    ...
)
```

Print out the time and we can see that it takes almost 5 hours.

```python
print("This finetuning takes %.2f seconds." % (end - start))
```

```
This finetuning takes 17779.09 seconds.
```

It does take a lot of time but let's look at its performance.
To evaluate the model we just trained, run:

```python
predictor.evaluate(test_path)
```

And the evaluation results are shown in command line output. The first value `0.740` is mAP in COCO standard, and the second one `0.932` is mAP in VOC standard (or mAP50). For more details about these metrics, see [COCO's evaluation guideline](https://cocodataset.org/#detection-eval).

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.740                                                                               
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.932                                                                               
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.819                                                                               
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.483                                                                               
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.617                                                                               
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.792                                                                               
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.569                                                                               
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.811                                                                               
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.827                                                                               
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.603                                                                               
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.754                                                                               
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.866  
```

Under this high performance finetune setting, it took 5 hours but reached `mAP50 = 0.932` on VOC!
For how to finetune faster,
see :ref:`sec_automm_detection_fast_finetune_coco`, where we finetuned a YOLOv3 model with
100 seconds and reached `mAP50 = 0.755` on VOC.

### Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

### Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.

### Citation

```
@article{DBLP:journals/corr/abs-2008-13367,
  author    = {Haoyang Zhang and
               Ying Wang and
               Feras Dayoub and
               Niko S{\"{u}}nderhauf},
  title     = {VarifocalNet: An IoU-aware Dense Object Detector},
  journal   = {CoRR},
  volume    = {abs/2008.13367},
  year      = {2020},
  url       = {https://arxiv.org/abs/2008.13367},
  eprinttype = {arXiv},
  eprint    = {2008.13367},
  timestamp = {Wed, 16 Sep 2020 11:20:03 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2008-13367.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
