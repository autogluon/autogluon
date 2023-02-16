# AutoMM Detection - Quick Start on a Tiny COCO Format Dataset
:label:`sec_automm_detection_quick_start_coco`

In this section, our goal is to fast finetune a pretrained model on a small dataset in COCO format, 
and evaluate on its test set. Both training and test sets are in COCO format.
See :ref:`sec_automm_detection_convert_to_coco` for how to convert other datasets to COCO format.

### Setting up the imports
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

### Downloading Data
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

### Creating the MultiModalPredictor
We select the "medium_quality" presets, which uses a YOLOX-small model pretrained on COCO dataset. This preset is fast to finetune or inference,
and easy to deploy. We also provide presets "high_quality" and "best quality", with higher performance but also slower.

```python .input
presets = "medium_quality"
```

We create the MultiModalPredictor with selected presets.
We need to specify the problem_type to `"object_detection"`,
and also provide a `sample_data_path` for the predictor to infer the catgories of the dataset.
Here we provide the `train_path`, and it also works using any other split of this dataset.
And we also provide a `path` to save the predictor. 
It will be saved to a automatically generated directory with timestamp under `AutogluonModels` if `path` is not specified.

```python .input
# Init predictor
import uuid

model_path = f"./tmp/{uuid.uuid4().hex}-quick_start_tutorial_temp_save"

predictor = MultiModalPredictor(
    problem_type="object_detection",
    sample_data_path=train_path,
    presets=presets,
    path=model_path,
)
```

### Finetuning the Model

Learning rate, number of epochs, and batch_size are included in the presets, and thus no need to specify.
Note that we use a two-stage learning rate option during finetuning by default,
and the model head will have 100x learning rate.
Using a two-stage learning rate with high learning rate only on head layers makes
the model converge faster during finetuning. It usually gives better performance as well,
especially on small datasets with hundreds or thousands of images.
We also compute the time of the fit process here for better understanding the speed.
We run it on a g4.2xlarge EC2 machine on AWS,
and part of the command outputs are shown below:

```python .input
start = time.time()
predictor.fit(train_path)  # Fit
train_end = time.time()
```

Notice that at the end of each progress bar, if the checkpoint at current stage is saved,
it prints the model's save path.
In this example, it's `./quick_start_tutorial_temp_save`.

Print out the time and we can see that it's fast!

```python .input
print("This finetuning takes %.2f seconds." % (train_end - start))
```

### Evaluation

To evaluate the model we just trained, run following code.

And the evaluation results are shown in command line output. 
The first line is mAP in COCO standard, and the second line is mAP in VOC standard (or mAP50). 
For more details about these metrics, see [COCO's evaluation guideline](https://cocodataset.org/#detection-eval).
Note that for presenting a fast finetuning we use presets "medium_quality", 
you could get better result on this dataset by simply using "high_quality" or "best_quality" presets, 
or customize your own model and hyperparameter settings: :ref:`sec_automm_customization`.

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
new_predictor = MultiModalPredictor.load(model_path)
new_predictor.set_num_gpus(1)
```

Evaluating the new predictor gives us exactly the same result:

```python .input
# Evaluate new predictor
new_predictor.evaluate(test_path)
```

For how to set the hyperparameters and finetune the model with higher performance, 
see :ref:`sec_automm_detection_high_ft_coco`.

### Inference
Now that we have gone through the model setup, finetuning, and evaluation, this section details the inference. 
Specifically, we layout the steps for using the model to make predictions and visualize the results.

To run inference on the entire test set, perform:

```python .input
pred = predictor.predict(test_path)
print(pred)
```
The output `pred` is a `pandas` `DataFrame` that has two columns, `image` and `bboxes`.

In `image`, each row contains the image path

In `bboxes`, each row is a list of dictionaries, each one representing a bounding box: `{"class": <predicted_class_name>, "bbox": [x1, y1, x2, y2], "score": <confidence_score>}`

Note that, by default, the `predictor.predict` does not save the detection results into a file.

To run inference and save results, run the following:
```python .input
pred = better_predictor.predict(test_path, save_results=True)
```
Here, we save `pred` into a `.txt` file, which exactly follows the same layout as in `pred`.
You can use a predictor initialized in anyway (i.e. finetuned predictor, predictor with pretrained model, etc.).
Here, we demonstrate using the `better_predictor` loaded previously.

### Visualizing Results
To run visualizations, ensure that you have `opencv` installed. If you haven't already, install `opencv` by running 
```python .input
!pip install opencv-python
```

To visualize the detection bounding boxes, run the following:
```python .input
from autogluon.multimodal.utils import Visualizer

conf_threshold = 0.4  # Specify a confidence threshold to filter out unwanted boxes
image_result = pred.iloc[30]

img_path = image_result.image  # Select an image to visualize

visualizer = Visualizer(img_path)  # Initialize the Visualizer
out = visualizer.draw_instance_predictions(image_result, conf_threshold=conf_threshold)  # Draw detections
visualized = out.get_image()  # Get the visualized image

from PIL import Image
from IPython.display import display
img = Image.fromarray(visualized, 'RGB')
display(img)
```

### Testing on Your Own Image
You can also download an image and run inference on that single image. The follow is an example:

Download the example image:
```python .input
from autogluon.multimodal import download
image_url = "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
test_image = download(image_url)
```

Run inference:

```python .input
pred_test_image = better_predictor.predict({"image": [test_image]})
print(pred_test_image)
```

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

### Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

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
