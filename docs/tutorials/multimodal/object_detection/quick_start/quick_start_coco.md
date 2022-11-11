# AutoMM Detection - Quick Start on a COCO Format Dataset
:label:`sec_automm_detection_fast_finetune_coco`

In this section, our goal is to fast finetune a pretrained model on a small dataset in COCO format, 
and evaluate on its test set. Both training and test sets are in COCO format.
See :ref:`sec_automm_detection_convert_to_coco` for how to convert other datasets to COCO format.

To start, let's import MultiModalPredictor:

```python
from autogluon.multimodal import MultiModalPredictor
```

And also import some other packages that will be used in this tutorial:

```python
import os
import time

from autogluon.core.utils.loaders import load_zip
```

We have the sample dataset ready in the cloud. Let's download it:

```python
zip_file = "s3://automl-mm-bench/object_detection_dataset/tiny_motorbike_coco.zip"
download_dir = "./tiny_motorbike_coco"

load_zip.unzip(zip_file, unzip_dir=download_dir)
data_dir = os.path.join(download_dir, "tiny_motorbike")
train_path = os.path.join(data_dir, "Annotations", "coco_trainval.json")
test_path = os.path.join(data_dir, "Annotations", "coco_test.json")
```

While using COCO format dataset, the input is the json annotation file of the dataset split.
In this example, `coco_trainval.json` is the annotation file of the train-and-validate split,
and `coco_test.json` is the annotation file of the test split.

We select the YOLOv3 with MobileNetV2 as backbone,
and input resolution is 320x320, pretrained on COCO dataset. With this setting, it is fast to finetune or inference,
and easy to deploy.
For more model choices, see :label:`sec_automm_detection_selecting_models`.
And we use all the GPUs (if any):

```python
checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
num_gpus = -1  # use all GPUs
```

We create the MultiModalPredictor with selected checkpoint name and number of GPUs.
We need to specify the problem_type to `"object_detection"`,
and also provide a `sample_data_path` for the predictor to infer the catgories of the dataset.
Here we provide the `train_path`, and it also works using any other split of this dataset.
And we also provide a `path` to save the predictor. 
It will be saved to a automatically generated directory with timestamp under `AutogluonModels` if `path` is not specified.

```python
# Init predictor
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
    sample_data_path=train_path,
    path="./quick_start_tutorial_temp_save",
)
```

We set the learning rate to be `2e-4`.
Note that we use a two-stage learning rate option during finetuning by default,
and the model head will have 100x learning rate.
Using a two-stage learning rate with high learning rate only on head layers makes
the model converge faster during finetuning. It usually gives better performance as well,
especially on small datasets with hundreds or thousands of images.
We also set the epoch to be 15 and batch_size to be 32.
For more information about how to tune those hyperparameters,
see :ref:`sec_automm_detection_tune_hyperparameters`.
We also compute the time of the fit process here for better understanding the speed.

```python
start = time.time()
# Fit
predictor.fit(
    train_path,
    hyperparameters={
        "optimization.learning_rate": 2e-4, # we use two stage and detection head has 100x lr
        "optimization.max_epochs": 15,
        "env.per_gpu_batch_size": 32,  # decrease it when model is large
    },
)
train_end = time.time()
```

We run it on a g5dn.12xlarge EC2 machine on AWS,
and part of the command outputs are shown below:

```
Epoch 0:  50%|█████████████████████████████                             | 2/4 [00:01<00:01,  1.25it/s, loss=1.65e+05, v_num=Epoch 0, global step 1: 'val_direct_loss' reached 31240.10156 (best 31240.10156), saving model to '/media/code/autogluon/examples/automm/object_detection/quick_start_tutorial_temp_save/epoch=0-step=1.ckpt' as top 1                                     
Epoch 0: 100%|██████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.66it/s, loss=8.57e+04, v_num=Epoch 0, global step 2: 'val_direct_loss' reached 19609.82031 (best 19609.82031), saving model to '/media/code/autogluon/examples/automm/object_detection/quick_start_tutorial_temp_save/epoch=0-step=2.ckpt' as top 1
Epoch 1:  50%|█████████████████████████████                             | 2/4 [00:01<00:01,  1.35it/s, loss=7.93e+04, v_num=Epoch 1, global step 3: 'val_direct_loss' reached 12368.74121 (best 12368.74121), saving model to '/media/code/autogluon/examples/automm/object_detection/quick_start_tutorial_temp_save/epoch=1-step=3.ckpt' as top 1
Epoch 1: 100%|██████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.68it/s, loss=5.99e+04, v_num=Epoch 1, global step 4: 'val_direct_loss' reached 7318.06152 (best 7318.06152), saving model to '/media/code/autogluon/examples/automm/object_detection/quick_start_tutorial_temp_save/epoch=1-step=4.ckpt' as top 1
Epoch 2:  50%|█████████████████████████████                             | 2/4 [00:01<00:01,  1.37it/s, loss=5.27e+04, v_num=Epoch 2, global step 5: 'val_direct_loss' reached 4694.17773 (best 4694.17773), saving model to '/media/code/autogluon/examples/automm/object_detection/quick_start_tutorial_temp_save/epoch=2-step=5.ckpt' as top 1
Epoch 2: 100%|██████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.74it/s, loss=4.41e+04, v_num=Epoch 2, global step 6: 'val_direct_loss' reached 2747.12524 (best 2747.12524), saving model to '/media/code/autogluon/examples/automm/object_detection/quick_start_tutorial_temp_save/epoch=2-step=6.ckpt' as top 1
Epoch 3:  50%|█████████████████████████████                             | 2/4 [00:01<00:01,  1.38it/s, loss=3.91e+04, v_num=Epoch 3, global step 7: 'val_direct_loss' reached 1862.42139 (best 1862.42139), saving model to '/media/code/autogluon/examples/automm/object_detection/quick_start_tutorial_temp_save/epoch=3-step=7.ckpt' as top 1
Epoch 3: 100%|██████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.74it/s, loss=3.42e+04, v_num=Epoch 3, global step 8: 'val_direct_loss' reached 1293.18774 (best 1293.18774), saving model to '/media/code/autogluon/examples/automm/object_detection/quick_start_tutorial_temp_save/epoch=3-step=8.ckpt' as top 1
Epoch 4:  50%|█████████████████████████████                             | 2/4 [00:01<00:01,  1.38it/s, loss=3.08e+04, v_num=Epoch 4, global step 9: 'val_direct_loss' reached 925.55371 (best 925.55371), saving model to '/media/code/autogluon/examples/automm/object_detection/quick_start_tutorial_temp_save/epoch=4-step=9.ckpt' as top 1
```

Notice that at the end of each progress bar, if the checkpoint at current stage is saved,
it prints the model's save path.
In this example, it's `./quick_start_tutorial_temp_save`.

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
see :ref:`sec_automm_detection_high_performance_finetune_coco`, where we finetuned a VFNet model with 
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
