# AutoMM Detection - Quick Start on a Tiny COCO Format Dataset
:label:`sec_automm_detection_quick_start_coco`

In this section, our goal is to fast finetune a pretrained model on a small dataset in COCO format, 
and evaluate on its test set. Both training and test sets are in COCO format.
See :ref:`sec_automm_detection_convert_to_coco` for how to convert other datasets to COCO format.

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
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
download_dir = "./tiny_motorbike_coco"

load_zip.unzip(zip_file, unzip_dir=download_dir)
data_dir = os.path.join(download_dir, "tiny_motorbike")
train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
```

While using COCO format dataset, the input is the json annotation file of the dataset split.
In this example, `trainval_cocoformat.json` is the annotation file of the train-and-validate split,
and `test_cocoformat.json` is the annotation file of the test split.

We select the YOLOv3 with MobileNetV2 as backbone,
and input resolution is 320x320, pretrained on COCO dataset. With this setting, it is fast to finetune or inference,
and easy to deploy.
For more model choices, see :label:`sec_automm_detection_selecting_models`.
And we use all the GPUs (if any):

```python .input
checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
num_gpus = -1  # use all GPUs
```

We create the MultiModalPredictor with selected checkpoint name and number of GPUs.
We need to specify the problem_type to `"object_detection"`,
and also provide a `sample_data_path` for the predictor to infer the catgories of the dataset.
Here we provide the `train_path`, and it also works using any other split of this dataset.
And we also provide a `path` to save the predictor. 
It will be saved to a automatically generated directory with timestamp under `AutogluonModels` if `path` is not specified.

```python .input
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
We run it on a g4.2xlarge EC2 machine on AWS,
and part of the command outputs are shown below:

```python .input
start = time.time()
# Fit
predictor.fit(
    train_path,
    hyperparameters={
        "optimization.learning_rate": 2e-4, # we use two stage and detection head has 100x lr
        "optimization.max_epochs": 30,
        "env.per_gpu_batch_size": 32,  # decrease it when model is large
    },
)
train_end = time.time()
```

Notice that at the end of each progress bar, if the checkpoint at current stage is saved,
it prints the model's save path.
In this example, it's `./quick_start_tutorial_temp_save`.

Print out the time and we can see that it's fast!

```python .input
print("This finetuning takes %.2f seconds." % (train_end - start))
```

To evaluate the model we just trained, run following code.

And the evaluation results are shown in command line output. 
The first line is mAP in COCO standard, and the second line is mAP in VOC standard (or mAP50). 
For more details about these metrics, see [COCO's evaluation guideline](https://cocodataset.org/#detection-eval).
Note that for presenting a fast finetuning we use 15 epochs, 
you could get better result on this dataset by simply increasing the epochs.

```python .input
predictor.evaluate(test_path)
eval_end = time.time()
```

Print out the evaluation time:

```python .input
print("The evaluation takes %.2f seconds." % (eval_end - train_end))
```

We can load a new predictor with previous save_path,
and we can also reset the number of GPUs to use if not all the devices are available:

```python .input
# Load and reset num_gpus
new_predictor = MultiModalPredictor.load("./quick_start_tutorial_temp_save")
new_predictor.set_num_gpus(1)
```

Evaluating the new predictor gives us exactly the same result:

```python .input
# Evaluate new predictor
new_predictor.evaluate(test_path)
```

If we set validation metric to `"map"` (Mean Average Precision), and max epochs to `50`, 
the predictor will have better performance with the same pretrained model (YOLOv3).
We trained it offline and uploaded to S3. To load and check the result:
```python .input
# Load Trained Predictor from S3
zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection/quick_start/AP50_433.zip"
download_dir = "./AP50_433"
load_zip.unzip(zip_file, unzip_dir=download_dir)
better_predictor = MultiModalPredictor.load("./AP50_433/quick_start_tutorial_temp_save")
better_predictor.set_num_gpus(1)

# Evaluate new predictor
better_predictor.evaluate(test_path)
```

For how to set those hyperparameters and finetune the model with higher performance, 
see :ref:`sec_automm_detection_high_performance_finetune_coco`.

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
