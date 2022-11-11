# AutoMM Detection - Inference with Pretrained VFNet on COCO Dataset
:label:`sec_automm_detection_infer_coco`

In this section, we show an example to run inference COCO dataset in COCO Format. 
Different from running evaluation, the purpose is to get detection results for potential down-stream tasks.
The model we use is the VFNet pretrained on COCO dataset.

## Prepare data
For running this tutorial, you should have COCO dataset prepared.
If you haven't already, head over to :label:`sec_automm_detection_prepare_coco17` to learn how to get COCO dataset.
This tutorial assumes that COCO dataset is under `~/data`, i.e. it is located in `~/data/coco17`

## Creating the `MultiModalPredictor`
To start, import MultiModalPredictor:
```{.python .input}
from autogluon.multimodal import MultiModalPredictor
```
### Use a pretrained model
You can download a pretrained model and construct a predictor with it. 
In this example, we use the VFNet with ResNext as backbone and Feature Pyramid Network (FPN) as neck.

```{.python .input}
checkpoint_name = "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco"
num_gpus = 1  # set to -1 to use all GPUs if available
```
You can also use other model by setting `checkpoint_name` to other names. 
Please refer to :ref: `selecting_models` for details about model selection.

As before, we create the MultiModalPredictor with selected checkpoint name and number of GPUs.
We also need to specify the `problem_type` to `"object_detection"`.

```{.python .input}
predictor = MultiModalPredictor(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": checkpoint_name,
        "env.num_gpus": num_gpus,
    },
    problem_type="object_detection",
)
```

### Use a finetuned model
You can also use a previously trained/finetuned predictor to run inference with.
First specify the predictor path, for example:
```{.python .input}
load_path = "./AutogluonModels/ag-20221104_185342"  # replace this with path to your desired predictor
```
Then load the predictor:
```{.python .input}
predictor = MultiModalPredictor.load(load_path)
```

## Setting up data

For COCO format data, we need to provide the path for the data split used for inference.

```{.python .input}
test_path = "~/data/coco17/annotations/instances_val2017.json"
```

## Running inference
To run inference, perform:

```{.python .input}
pred = predictor.predict(test_path)
```
By default, the `predictor.predict` does not save the detection results into a file.

To run inference and save results, run the following:
```{.python .input}
pred = predictor.predict(test_path, save_results=True)
```
Currently, we convert the results to a pandas `DataFrame` and save into a `.txt` file.
The `.txt` file has two columns, `image` and `bboxes`, where
- in `image`, each row contains the image path
- in `bboxes`, each row is a list of dictionaries, each one representing a bounding box: 
  - `{"class": <predicted_class_name>, "bbox": [x1, y1, x2, y2], "score": <confidence_score>}`

## Reading results
By default, the returned value `pred` is a `list` and has the following dimensions:
```
[num_images, num_total_classes, num_detections_per_class, 5]
```

where 
- `num_images` is the number of images used to run inference on. 
- `num_total_classes` is the total number of classes depending on the model specified in `predictor`. In this example, `num_total_classes = 80` for the VFNet model pretrained on COCO dataset. To get all available classes in the predictor, run `predictor.get_predictor_classes()`
- `num_detections_per_class` is the number of detections under each class. Note that this number can vary across different classes.
- The last dimension contains the bounding box information, which follows `[x1, y1, x2, y2, score]` format. `x1, y1` are the top left corner of the bounding box, and `x2, y2` are the bottom right corner. `score` is the confidence score of the prediction

example code to examine bounding box information:

```{.python .input}
detection_classes = predictor.get_predictor_classes()
idx2classname = {idx: classname for (idx, classname) in enumerate(detection_classes)}
for i, image_pred in enumerate(pred):
    print("result for image {}".format(i))
    for j, per_cls_bboxes in enumerate(image_pred):
        classname = idx2classname[j]
        for bbox in per_cls_bboxes:
            ## bbox = [x1, y1, x2, y2, conf_score]
            print("bbox: {}, class: {}, score: {}".format(bbox[:4], classname, bbox[4]))
```
If you prefer to get the results in `pd.DataFrame` format, run the following:
```{.python .input}
pred_df = predictor.predict(test_path, as_pandas=True)
```

Similar to the `.txt` file, the `pred_df` also has two columns, `image` and `bboxes`, where
- in `image`, each row contains the image path
- in `bboxes`, each row is a list of dictionaries, each one representing a bounding box: 
  - `{"class": <predicted_class_name>, "bbox": [x1, y1, x2, y2], "score": <confidence_score>}`

## Visualizing Results
To run visualizations, ensure that you have `opencv` installed. If you haven't already, install `opencv` by running 
```{.python .input}
pip install opencv-python
```

To visualize the detection bounding boxes, run the following:
```{.python .input}
from autogluon.multimodal.utils import from_coco_or_voc, visualize_detection
import matplotlib.pyplot as plt

conf_threshold = 0.4  # Specify a confidence threshold to filter out unwanted boxes
visualization_result_dir = "~/data/coco17/visualizations"  # Specify a directory to save visualized images.
df = from_coco_or_voc(test_path)[:10][["image"]]

pred = predictor.predict(df)

visualized = visualize_detection(
    pred=pred,
    data=df,
    detection_classes=predictor.get_predictor_classes(),
    conf_threshold=conf_threshold,
    visualization_result_dir=visualization_result_dir,
)

plt.imshow(visualized[0][:, : ,::-1])  # shows the first image with bounding box
```
Note that we took 10 images to visualize for this example. 
Please consider your storage situation when deciding the number of images to visualize.

### Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

### Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.

### Citation
```
@inproceedings{zhang2021varifocalnet,
  title={Varifocalnet: An iou-aware dense object detector},
  author={Zhang, Haoyang and Wang, Ying and Dayoub, Feras and Sunderhauf, Niko},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8514--8523},
  year={2021}
}
```