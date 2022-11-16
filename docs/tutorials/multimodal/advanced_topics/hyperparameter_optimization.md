# Hyperparameter Optimization in AutoMM

Hyperparameter optimization (HPO) is a method that helps solve the challenge of tuning hyperparameters of machine learning models. ML algorithms have multiple complex hyperparameters that generate an enormous search space, and the search space in deep learning methods is even larger than traditional ML algorithms. Tuning on a massive search space is a tough challenge, but AutoMM provides various options for you to guide the fitting process based on your domain knowledge and the constraint on computing resources.

## Create Image Dataset

In this tutorial, we are going to again use the subset of the [Shopee-IET dataset](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/data) from Kaggle for demonstration purpose. Each image contains a clothing item and the corresponding label specifies its clothing category. Our subset of the data contains the following possible labels: `BabyPants`, `BabyShirt`, `womencasualshoes`, `womenchiffontop`.

We can load a dataset by downloading a url data automatically:

```{.python .input}
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

from autogluon.multimodal.utils.misc import shopee_dataset
download_dir = './ag_automm_tutorial_hpo'
train_data, test_data = shopee_dataset(download_dir)
train_data = train_data.sample(frac=0.5)
print(train_data)
```

There are in total 400 data points in this dataset. The `image` column stores the path to the actual image, and the `label` column stands for the label class. 


## The Regular Model Fitting

Recall that if we are to use the default settings predefined by Autogluon, we can simply fit the model using `MultiModalPredictor` with three lines of code:


```{.python .input}
from autogluon.multimodal import MultiModalPredictor
predictor_regular = MultiModalPredictor(label="label")
start_time = datetime.now()
predictor_regular.fit(
    train_data=train_data,
    hyperparameters = {"model.timm_image.checkpoint_name": "ghostnet_100"}
)
end_time = datetime.now()
elapsed_seconds = (end_time - start_time).total_seconds()
elapsed_min = divmod(elapsed_seconds, 60)
print("Total fitting time: ", f"{int(elapsed_min[0])}m{int(elapsed_min[1])}s")
```

Let's check out the test accuracy of the fitted model:

```{.python .input}
scores = predictor_regular.evaluate(test_data, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores["accuracy"])
```


## Use HPO During Model Fitting

If you would like more control over the fitting process, you can specify various options for hyperparameter optimizations(HPO) in `MultiModalPredictor` by simply adding more options in `hyperparameter` and `hyperparameter_tune_kwargs`.

There are a few options we can have in MultiModalPredictor. We use [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) `tune` library in the backend, so we need to pass in a [Tune search space](https://docs.ray.io/en/latest/tune/api_docs/search_space.html) or an [AutoGluon search space](https://auto.gluon.ai/dev/tutorials/course/core.html) which will be converted to Tune search space.

1. Defining the search space of various `hyperparameter` values for the training of neural networks:

<ul>
```
hyperparameters = {
        "optimization.learning_rate": tune.uniform(0.00005, 0.005),
        "optimization.optim_type": tune.choice(["adamw", "sgd"]),
        "optimization.max_epochs": tune.choice(["10", "20"]), 
        "model.timm_image.checkpoint_name": tune.choice(["swin_base_patch4_window7_224", "convnext_base_in22ft1k"])
        }
```

This is an example but not an exhaustive list. You can find the full supported list in [Customize AutoMM](https://auto.gluon.ai/stable/tutorials/multimodal/customization.html#sec-automm-customization)
</ul>
    
2. Defining the search strategy for HPO with `hyperparameter_tune_kwargs`. You can pass in a string or initialize a `ray.tune.schedulers.TrialScheduler` object.

<ul>
a. Specifying how to search through your chosen hyperparameter space (supports `random` and `bayes`):
    
```
"searcher": "bayes"
```
</ul>

<ul>
b. Specifying how to schedule jobs to train a network under a particular hyperparameter configuration (supports `FIFO` and `ASHA`):

```            
"scheduler": "ASHA"
```
</ul>

<ul>
c. Number of trials you would like to carry out HPO:
            
```
"num_trials": 20
```
</ul>

Let's work on HPO with combinations of different learning rates and backbone models:

```{.python .input}
from ray import tune

predictor_hpo = MultiModalPredictor(label="label")

hyperparameters = {
            "optimization.learning_rate": tune.uniform(0.00005, 0.001),
            "model.timm_image.checkpoint_name": tune.choice(["ghostnet_100",
                                                             "mobilenetv3_large_100"])
}
hyperparameter_tune_kwargs = {
    "searcher": "bayes", # random
    "scheduler": "ASHA",
    "num_trials": 2,
}
start_time_hpo = datetime.now()
predictor_hpo.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )
end_time_hpo = datetime.now()
elapsed_seconds_hpo = (end_time_hpo - start_time_hpo).total_seconds()
elapsed_min_hpo = divmod(elapsed_seconds_hpo, 60)
print("Total fitting time: ", f"{int(elapsed_min_hpo[0])}m{int(elapsed_min_hpo[1])}s")
```

Let's check out the test accuracy of the fitted model after HPO:

```{.python .input}
scores_hpo = predictor_hpo.evaluate(test_data, metrics=["accuracy"])
print('Top-1 test acc: %.3f' % scores_hpo["accuracy"])
```

From the training log, you should be able to see the current best trial as below:

```
Current best trial: 47aef96a with val_accuracy=0.862500011920929 and parameters={'optimization.learning_rate': 0.0007195214018085505, 'model.timm_image.checkpoint_name': 'ghostnet_100'}
```
After our simple 2-trial HPO run, we got a better test accuracy, by searching different learning rates and models, compared to the out-of-box solution provided in the previous section. HPO helps select the combination of hyperparameters with highest validation accuracy. 

## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
