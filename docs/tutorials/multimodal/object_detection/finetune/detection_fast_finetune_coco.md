# AutoMM Detection - Fast Finetune on COCO Format Dataset
:label:`sec_automm_detection_fast_ft_coco`

![Pothole Dataset](https://storage.googleapis.com/kagglesdsdata/datasets/702771/1228192/images/potholes101.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20221124%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20221124T064243Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=7ea47c3f7dca097792c950e0ad33114b7bd22b4d8a2882270a6c278870c24674f4aa4572a124c82068b30e56b184f8ab8a57e05993c146fd2defcd565db5b21d80d1e7eebfdf39c24344e8a8f9b01c0382252e36f8d1a09e0be20681b8b8e986f678316da3a7f6216e0b5195a78126133e997bb77f92666cd90285abdac6ab8839a33c22a18fbd19dec3a044a4ac9018c977a84894926a252094c39a7e3df4fba320ddd462e542a8cc7311e16cb8e66e9eef4cf1712bca1ee61780461039c79c5c5ab58f14400cb54ff9da34d706fbe0184cb0a3ed9a0fae13e130b8d86d23782362e72a3b34c0ffc08a20b26349bba0cc7de7cf36b31a8aa4e475d58a7c9dae)
:width:`500px`

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

We select the YOLOv3 with MobileNetV2 as backbone,
and input resolution is 320x320, pretrained on COCO dataset. With this setting, it is fast to finetune or inference,
and easy to deploy.
And we use all the GPUs (if any):

```python .input
checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
num_gpus = -1  # use all GPUs
```

We create the MultiModalPredictor with selected checkpoint name and number of GPUs.
We need to specify the problem_type to `"object_detection"`,
and also provide a `sample_data_path` for the predictor to infer the catgories of the dataset.
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

We set the learning rate to be `2e-4`.
Note that we use a two-stage learning rate option during finetuning by default,
and the model head will have 100x learning rate.
Using a two-stage learning rate with high learning rate only on head layers makes
the model converge faster during finetuning. It usually gives better performance as well,
especially on small datasets with hundreds or thousands of images.
We also set the epoch to be 30 for fast finetuning and batch_size to be 32.
We also compute the time of the fit process here for better understanding the speed.

```python .input
import time
start = time.time()
predictor.fit(
    train_path,
    hyperparameters={
        "optimization.learning_rate": 2e-4, # we use two stage and detection head has 100x lr
        "optimization.max_epochs": 30,
        "env.per_gpu_batch_size": 32,  # decrease it when model is large
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
conf_threshold = 0.1  # Specify a confidence threshold to filter out unwanted boxes
visualization_result_dir = "./"  # Use the pwd as result dir to save the visualized image
visualized = visualize_detection(
    pred=pred[4:5],
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
