# How to use FocalLoss
:label:`sec_automm_focal_loss`

In this tutorial, we introduce how to use `FocalLoss` from the AutoMM package for balanced training.
FocalLoss is first introduced in this [Paper](https://www.microsoft.com/en-us/research/uploads/prod/2021/10/model-size-graph.jpg)
and can be used for balancing hard/easy samples as well as un-even sample distribution among classes. This tutorial demonstrates how to use `FocalLoss`.

## Create Dataset
We use the shopee dataset for demonstration in this tutorial.
```{.python .input}
from autogluon.multimodal.utils.misc import shopee_dataset

download_dir = "./ag_automm_tutorial_imgcls_focalloss"
train_data, test_data = shopee_dataset(download_dir)
```

## Create and train `MultiModalPredictor` with `FocalLoss`
We specify the model to use `FocalLoss` by setting the `"optimization.loss_function"` to `"focal_loss"`.
There are also three other optional parameters you can set.

`optimization.focal_loss.alpha` - a list of floats which is the per-class loss weight that can be used to balance un-even sample distribution across classes.
Note that the `len` of the list ***must*** match the total number of classes in the training dataset.

`optimization.focal_loss.gamma` - float which controls how much to focus on the hard samples. Larger value means more focus on the hard samples.

`optimization.focal_loss.reduction` - how to aggregate the loss value. Can only take `"mean"` or `"sum"` for now.

```{.python .input}
import uuid
from autogluon.multimodal import MultiModalPredictor

model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"

predictor = MultiModalPredictor(label="label", problem_type="multiclass", path=model_path)

predictor.fit(
    hyperparameters={
        "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_gpus": 1,
        "optimization.loss_function": "focal_loss",
        "optimization.focal_loss.alpha": [1, 0.25, 0.35, 0.16],  # shopee dataset has 4 classes.
        "optimization.focal_loss.gamma": 2.5,
        "optimization.focal_loss.reduction": "mean",
        "optimization.max_epochs": 1,
    },
    train_data=train_data,
    time_limit=30,  # seconds
)  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

predictor.evaluate(test_data, metrics=["acc"])
```


