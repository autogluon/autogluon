Step-by-step on creating new examples
===

## Step 1: Installation

    python setup.py develop

## Step 2: Create new example

    cd examples
    
Create examples using AutoGluon API following `autogluon_beginner_image_classification_cifar10.py`

For example: `autogluon_beginner_text_classification_yelp.py`
    
    
## Step 3: Create new task

    cd task/
    
Create new task folder, e.g.,
    
    mkdir text_classification

Then follow image_classification folder to create:

    -`core.py` contains `fit`
    -`dataset.py` contains `Dataset`
    -`model_zoo.py` contains `models` for the task, this should depend on gluoncv and gluonnlp.
    -`pipeline.py` contains training logic for the task.
    
    
## Example: Image classification example

You can run the example

    cd examples
    python autogluon_beginner_image_classification_cifar10.py
    python autogluon_advanced_image_classification_cifar10.py
