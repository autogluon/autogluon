# AutoMM Detection - High Performance Finetune on COCO Format Dataset
:label:`sec_automm_detection_high_ft_coco`

In this section, our goal is to fast finetune and evaluate a pretrained model 
on [Pothole dataset](https://www.kaggle.com/datasets/andrewmvd/pothole-detection) in COCO format.
Pothole is a single object, i.e. `pothole`, detection dataset, containing 665 images with bounding box annotations
for the creation of detection models and can work as POC/POV for the maintenance of roads.
See :ref:`sec_automm_detection_prepare_voc` for how to prepare Pothole dataset.

To start, let's import MultiModalPredictor:

```python .input
from autogluon.multimodal import MultiModalPredictor
```

Make sure `mmcv-full` and `mmdet` are installed:
```python .input
!mim install mmcv-full
!pip install mmdet
```

And also import some other packages that will be used in this tutorial:

```python .input
import os
import time

from autogluon.core.utils.loaders import load_zip
```

We have the sample dataset ready in the cloud. Let's download it:

```python .input
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection/dataset/pothole.zip"
download_dir = "./pothole"

load_zip.unzip(zip_file, unzip_dir=download_dir)
data_dir = os.path.join(download_dir, "pothole")
train_path = os.path.join(data_dir, "Annotations", "usersplit_train_cocoformat.json")
val_path = os.path.join(data_dir, "Annotations", "usersplit_val_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "usersplit_test_cocoformat.json")
```

While using COCO format dataset, the input is the json annotation file of the dataset split.
In this example, `usersplit_train_cocoformat.json` is the annotation file of the train split.
`usersplit_val_cocoformat.json` is the annotation file of the validation split.
And `usersplit_test_cocoformat.json` is the annotation file of the test split.

We select the VFNet with ResNet-50 as backbone, Feature Pyramid Network (FPN) as neck,
and input resolution is 640x640, pretrained on COCO dataset.
*(The neck of the object detector refers to the additional layers existing between the backbone and the head. 
Their role is to collect feature maps from different stages.)*
With this setting, it sacrifices training and inference time,
and also requires much more GPU memory,
but the performance is high. 

We use `val_metric = map`, i.e., mean average precision in COCO standard as our validation metric.
In previous section :ref:`sec_automm_detection_fast_ft_coco`,
we did not specify the validation metric and by default the validation loss is used as validation metric.
Using validation loss is much faster but using mean average precision gives the best performance.

And we use all the GPUs (if any):

```python .input
checkpoint_name = "vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco"
num_gpus = -1  # use all GPUs
val_metric = "map"
```

We create the MultiModalPredictor with selected checkpoint name, val_metric, and number of GPUs.
We need to specify the problem_type to `"object_detection"`,
and also provide a `sample_data_path` for the predictor to infer the catgories of the dataset.
Here we provide the `train_path`, and it also works using any other split of this dataset.

```python .input
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

We used `1e-4` for Yolo V3 in previous tutorial, 
but set the learning rate to be `5e-6` here, 
because larger model always prefers smaller learning rate.
Note that we use a two-stage learning rate option during finetuning by default,
and the model head will have 100x learning rate.
Using a two-stage learning rate with high learning rate only on head layers makes
the model converge faster during finetuning. It usually gives better performance as well,
especially on small datasets with hundreds or thousands of images.
We also set the batch_size to be 2, because this model is too huge to run with larger batch size.
We also compute the time of the fit process here for better understanding the speed.
We only set the number of epochs to be 1 for a quick demonstration, 
and to seriously finetune the model on this dataset we will need to set this to 20 or more.

```python .input
import time
start = time.time()
predictor.fit(
    train_path,
    hyperparameters={
        "optimization.learning_rate": 5e-6, # we use two stage and detection head has 100x lr
        "optimization.max_epochs": 1,
        "env.per_gpu_batch_size": 2,  # decrease it when model is large
    },
)
end = time.time()
```

Print out the time and we can see that it takes a long time even for one epoch.

```python .input
print("This finetuning takes %.2f seconds." % (end - start))
```

To get a model with reasonable performance, you will need to finetune the model with more epochs.
We set `max_epochs` to 50 and trained a model offline. And we uploaded it to AWS S3. 
To load and check the result:

```python .input
# Load Trained Predictor from S3
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection/checkpoints/pothole_AP50_718.zip"
download_dir = "./pothole_AP50_718.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)
better_predictor = MultiModalPredictor.load("./pothole_AP50_718/AutogluonModels/ag-20221123_021130")
better_predictor.set_num_gpus(1)

# Evaluate new predictor
better_predictor.evaluate(test_path)
```

Under this high performance finetune setting, it took a long time to train but reached `mAP50 = 0.718`!
For how to finetune faster,
see :ref:`sec_automm_detection_fast_ft_coco`, where we finetuned a YOLOv3 model with lower
performance but much faster.

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
