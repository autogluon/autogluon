# AutoMM Detection - Load A Trained Detector
:label:`sec_automm_detection_load`

In this section, our goal is to load and evaluate the predictor trained previously.

To start, let's import MultiModalPredictor:

```python
from autogluon.multimodal import MultiModalPredictor
```

We'll use the predictor we finetuned in previous tutorial :ref:`sec_automm_detection_fast_ft_coco`.
Note that you should have a different `load_path`. For test, we still use VOC2007 test set in COCO format.

```python
load_path = "./AutogluonModels/ag-20221104_185342"
test_path = "./VOCdevkit/VOC2007/Annotations/test_cocoformat.json"
```

We load the MultiModalPredictor in one line, with all previous settings recovered.

```python
predictor = MultiModalPredictor.load(load_path)
```

In some cases we may want to use different number of gpus for the predictor.
For example, here we used 4 gpus while training the preditor but there is only one available at this point.
Then we can reset the number of gpus by:

```python
predictor.set_num_gpus(num_gpus=1)
```

To evaluate the predictor we loaded, run:

```python
predictor.evaluate(test_path)
```

And the evaluation results are shown in command line output.
It's exactly the same as we got in previous training.

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

After the predictor is loaded, you can do the evaluation, prediction,
or finetune it on another dataset. But note the loaded predictor is saved at the same path
and the save operation in finetuning will overwrite the previous checkpoint.

### Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

### Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
