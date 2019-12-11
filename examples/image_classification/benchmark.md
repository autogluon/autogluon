# Documents for Kaggle Benchmark

This tutorial demonstrates how to use AutoGluon with your own custom datasets.
As an example, we use a dataset from Kaggle to show the required steps to format image data properly for AutoGluon.

## Step 1: Organizing the dataset into proper directories

After completing this step, you will have the following directory structure on your machine:

```
   XXXX/
    ├── class1/
    ├── class2/
    ├── class3/
    ├── ...
```

Here `XXXX` is a folder containing the raw images categorized into classes. For example, subfolder `class1` contains all images that belong to the first class, `class2` contains all images belonging to the second class, etc. 
We generally recommend at least 100 training images per class for reasonable classification performance, but this might depend on the type of images in your specific use-case.

Under each class, the following image formats are supported when training your model:

- JPG
- JPEG
- PNG

In the same dataset, all the images should be in the same format. Note that in image classification, we do not require that all images have the same resolution.

You will need to organize your dataset into the above directory structure before using AutoGluon.
Below, we demonstrate how to construct this organization for a Kaggle dataset.

### Example: Kaggle dataset

Kaggle is a popular machine learning competition platform and contains lots of
datasets for different machine learning tasks including image classification.
If you don't have Kaggle account, please register one at [Kaggle](https://www.kaggle.com/). 
Then, please follow the [Kaggle installation](https://github.com/Kaggle/kaggle-api/) to obtain access to Kaggle's data downloading API.

To find image classification datasets in Kaggle, let's go to [Kaggle](https://www.kaggle.com/) 
and search using keyword `image classification` either under `Datasets` or `Competitions`.

For example, we propose six datasets end to end training process in `Competitions` as follow:
[1 Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition),
[2 Aerial Cactus](https://www.kaggle.com/c/aerial-cactus-identification),
[3 Plant Seedlings](https://www.kaggle.com/c/plant-seedlings-classification),
[4 Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring), 
[5 Dog Breed](https://www.kaggle.com/c/dog-breed-identification),
[6 Shopee-IET](https://www.kaggle.com/c/shopee-iet-machine-learning-competition)

Please make sure to click the button of "I Understand and Accept" before downloading the data.

Downloading this script to your machine for directory structure.
```
import autogluon as ag
ag.download('https://raw.githubusercontent.com/aptsunny/AutoGluonWebdata/master/download_dataset.sh')
```

An example shell script to download the dataset to `./data/XXXX/` can be found here: [download_dataset.sh](https://raw.githubusercontent.com/aptsunny/AutoGluonWebdata/master/download_dataset.sh).

Run it with the dataset you want:
```
!sh download_shopeeiet.sh <cats\aerial\plant\dog\shopee>
```

Downloading this script to your machine for organize messy training data.
```
import autogluon as ag
ag.download('https://raw.githubusercontent.com/aptsunny/AutoGluonWebdata/master/data_processing.py')
```
Sometimes dataset needs additional data preprocessing by Script [data_processing](https://github.com/aptsunny/AutoGluonWebdata/blob/master/data_processing.py).
```
  data
    ├──XXXX/train
    ├         ├── AAAAAAAA
    ├         ├── BBBBBBBB
    ├──XXXX/test
    ├         ├── ...
    ├──XXXX/data_processing.py

python data_processing.py --dataset <aerial\dog\>
```

Finally, we have the desired directory structure under `./data/XXXX/train/`, which in this case looks as follows:

```
  data
    ├──XXXX/train
    ├         ├── AAAAAAAA
    ├         ├── BBBBBBBB
    ├         ├── CCCCCCCC
    ├         ├── ...
    ├──XXXX/test
    ├         ├── ...
    ├
    ├
    ├──ZZZZ/train
    ├         ├── AAAAAAAA
    ├         ├── BBBBBBBB
    ├         ├── CCCCCCCC
    ├         ├── ...
    ├──ZZZZ/test
              ├── ...
```

## Step 2: Use AutoGluon fit to generate a classification model

Now that we have a `Dataset` object, we can use AutoGluon's default configuration to obtain an image classification model using the [`fit`](/api/autogluon.task.html#autogluon.task.ImageClassification.fit) function.

Run `benchmark.py` script with different dataset:

`python autogluon/examples/image_classification/benchmark.py --data-dir ./data/ --dataset (XXXX)`

`XXXX` can be any one of the options below：

``` 
dogs-vs-cats-redux-kernels-edition
aerial-cactus-identification
plant-seedlings-classification
fisheries_Monitoring
dog-breed-identification
shopee-iet-machine-learning-competition
```

## Step 3:  fit to generate a classification model

Bag of tricks are used on image classification dataset.

Customize parameter configuration as follow:
```
            lr_config = {'lr_mode': 'cosine',
                         'lr_decay': 0.1,
                         'lr_decay_period': 0,
                         'lr_decay_epoch': '10,20,30',
                         'warmup_lr': 0.0,
                         'warmup_epochs': 5},
            tricks = {'last_gamma': True,
                      'use_pretrained': True,
                      'use_se': False,
                      'mixup': False,  
                      'mixup_alpha': 0.2,
                      'mixup_off_epoch': 0,
                      'label_smoothing': True,
                      'no_wd': True,
                      'teacher_name': None,
                      'temperature': 20.0,
                      'hard_weight': 0.5,
                      'batch_norm': False,
                      'use_gn': False}
```

## Step 4: Submit test predictions to Kaggle

If you wish to upload the model's predictions to Kaggle, here is how to convert them into a format suitable for a submission into the Kaggle competition:

This produces a submission file located at: `./data/XXXX/submission.csv`.

To see an example submission, check out `sample submission.csv` at this link: https://www.kaggle.com/c/XXXX/data.

To make your own submission, 

Run the command `kaggle competitions submit -c <XXX> -f submission.csv -m "Message"` or
click [Submission](https://www.kaggle.com/c/shopee-iet-machine-learning-competition/submit)
and then follow the steps in the submission page (upload submission file, describe the submission,
and click the `Make Submission` button). Let's see how your model fares in this competition!



    

