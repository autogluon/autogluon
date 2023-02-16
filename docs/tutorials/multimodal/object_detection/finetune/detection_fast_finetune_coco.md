# AutoMM Detection - Fast Finetune on COCO Format Dataset
:label:`sec_automm_detection_fast_ft_coco`

![Pothole Dataset](https://automl-mm-bench.s3.amazonaws.com/object_detection/example_image/pothole144_gt.jpg)
:width:`500px`

In this section, our goal is to fast finetune and evaluate a pretrained model 
on [Pothole dataset](https://www.kaggle.com/datasets/andrewmvd/pothole-detection) in COCO format.
Pothole is a single object, i.e. `pothole`, detection dataset, containing 665 images with bounding box annotations
for the creation of detection models and can work as POC/POV for the maintenance of roads.
See :ref:`sec_automm_detection_prepare_pothole` for how to prepare Pothole dataset.

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

We select the YOLOX-small model pretrained on COCO dataset. With this setting, it is fast to finetune or inference,
and easy to deploy. Note that you can use a YOLOX-large by setting the `checkpoint_name` to `"yolox_l_8x8_300e_coco"` for better performance (but slower speed).
Note that you may need to change the learning_rate and per_gpu_batch_size for a different model.
An easier way is to use our predefined presets `"medium_quality"`, `"high_quality"`, or `"best_quality"`.
For more about using presets, see :ref:`sec_automm_detection_quick_start_coco`.
And we use all the GPUs (if any):

```python .input
checkpoint_name = "yolox_s_8x8_300e_coco"
num_gpus = -1  # use all GPUs
```

We create the MultiModalPredictor with selected checkpoint name and number of GPUs.
We need to specify the problem_type to `"object_detection"`,
and also provide a `sample_data_path` for the predictor to infer the categories of the dataset.
Here we provide the `train_path`, and it also works using any other split of this dataset.

```python .input
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
    sample_data_path=train_path,
)
```

We set the learning rate to be `1e-4`.
Note that we use a two-stage learning rate option during finetuning by default,
and the model head will have 100x learning rate.
Using a two-stage learning rate with high learning rate only on head layers makes
the model converge faster during finetuning. It usually gives better performance as well,
especially on small datasets with hundreds or thousands of images.
We set batch size to be 16, and you can increase or decrease the batch size based on your available GPU memory.
We set max number of epochs to 30, number of validation check per interval to 1.0, 
and validation check per n epochs to 3 for fast finetuning.
We also compute the time of the fit process here for better understanding the speed.

```python .input
import time
start = time.time()
predictor.fit(
    train_path,
    tuning_data=val_path,
    hyperparameters={
        "optimization.learning_rate": 1e-4,  # we use two stage and detection head has 100x lr
        "env.per_gpu_batch_size": 16,  # decrease it when model is large
        "optimization.max_epochs": 30,  # max number of training epochs, note that we may early stop before this based on validation setting
        "optimization.val_check_interval": 1.0,  # Do 1 validation each epoch
        "optimization.check_val_every_n_epoch": 3,  # Do 1 validation each 3 epochs
        "optimization.patience": 3,  # Early stop after 3 consective validations are not the best
    },
)
end = time.time()
```

Print out the time and we can see that it's fast!

```python .input
print("This finetuning takes %.2f seconds." % (end - start))
```

To evaluate the model we just trained, run:

```python .input
predictor.evaluate(test_path)
```

Note that you can also use our predefined presets `"medium_quality"` to do the exact same thing with following code script:

```python
predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets="medium_quality",
)
predictor.fit(train_path, tuning_data=val_path)
predictor.evaluate(test_path)
```

For more about using presets, see :ref:`sec_automm_detection_quick_start_coco`.

And the evaluation results are shown in command line output. 
The first value is mAP in COCO standard, and the second one is mAP in VOC standard (or mAP50). 
For more details about these metrics, see [COCO's evaluation guideline](https://cocodataset.org/#detection-eval).

We can get the prediction on test set:

```python .input
pred = predictor.predict(test_path)
```

Let's also visualize the prediction result:

```python .input
!pip install opencv-python
```

```python .input
from autogluon.multimodal.utils import visualize_detection
conf_threshold = 0.25  # Specify a confidence threshold to filter out unwanted boxes
visualization_result_dir = "./"  # Use the pwd as result dir to save the visualized image
visualized = visualize_detection(
    pred=pred[12:13],
    detection_classes=predictor.get_predictor_classes(),
    conf_threshold=conf_threshold,
    visualization_result_dir=visualization_result_dir,
)
from PIL import Image
from IPython.display import display
img = Image.fromarray(visualized[0][:, :, ::-1], 'RGB')
display(img)
```

Under this fast finetune setting, we reached a good mAP number on a new dataset with a few hundred seconds!
For how to finetune with higher performance,
see :ref:`sec_automm_detection_high_ft_coco`, where we finetuned a VFNet model with 
5 hours and reached `mAP = 0.450, mAP50 = 0.718` on this dataset.

### Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

### Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.

### Citation
```
@article{DBLP:journals/corr/abs-2107-08430,
  author    = {Zheng Ge and
               Songtao Liu and
               Feng Wang and
               Zeming Li and
               Jian Sun},
  title     = {{YOLOX:} Exceeding {YOLO} Series in 2021},
  journal   = {CoRR},
  volume    = {abs/2107.08430},
  year      = {2021},
  url       = {https://arxiv.org/abs/2107.08430},
  eprinttype = {arXiv},
  eprint    = {2107.08430},
  timestamp = {Tue, 05 Apr 2022 14:09:44 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2107-08430.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org},
}
```
