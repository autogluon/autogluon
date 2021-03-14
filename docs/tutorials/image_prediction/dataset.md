# Image Prediction - Properly load any type of data as ImagePredictor Dataset
:label:`sec_imgdataset`

Preparing the dataset for ImagePredictor is not difficult at all, however, we'd like to introduce the
recommended ways to initialize the dataset so you will have smoother experience using `autogluon.vision.ImagePredictor`.

There are generally three ways to load a dataset for ImagePredictor:
- Load a csv file or construct your own `DataFrame` with `image` and `label` columns
- Load a image folder directly with `ImagePredictor.Dataset`
- Convert a list of images to dataset directly with `ImagePredictor.Dataset`

We will go through these four methods one by one. First of all, let's import it

```{.python .input}
%matplotlib inline
import autogluon.core as ag
from autogluon.vision import ImagePredictor
import pandas as pd
```

## Load a csv file or construct a DataFrame object

We use a csv file from PetFinder competition as an example. You may use any tabular data as long as you can
create `image`(absolute or relative paths to images) and `label`(category for each image) columns.

```{.python .input}
csv_file = ag.utils.download('https://autogluon.s3-us-west-2.amazonaws.com/datasets/petfinder_example.csv')
df = pd.read_csv(csv_file)
df.head()
```

If the image paths are not relative to current working directory, you may use the helper function to prepend prefix for each image, using absolute paths can reduce the chance of OSError happening to file access:

```{.python .input}
df = ImagePredictor.Dataset.from_csv(csv_file, root='/home/ubuntu')
df.head()
```

Or you can perform the correction by yourself:

```{.python .input}
import os
df['image'] = df['image'].apply(lambda x: os.path.join('/home/ubuntu', x))
df.head()
```

Otherwise you may use the `DataFrame` as-is, `ImagePredictor` will apply auto conversion during `fit` to ensure other metadata is available for training. You can have multiple columns in the `DataFrame`, `ImagePredictor` only cares about `image` and `label` columns during training.

## Load an image directory

It's pretty common that sometimes you only have a folder of images, organized by the category names. Recursively loop through images is boring. You can use `ImagePredictor.Dataset.from_folders` or `ImagePredictor.Dataset.from_folder` to avoid implementing recursive search.

The difference between `from_folders` and `from_folder` is the targeting folder structure.
If you have a folder with splits, e.g., `train`, `test`, like:

- root/train/car/0001.jpg
- root/train/car/xxxa.jpg
- root/val/bus/123.png
- root/test/bus/023.jpg

Then you can load the splits with `from_folders`:

```{.python .input}
train_data, _, test_data = ImagePredictor.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip', train='train', test='test')
print('train #', len(train_data), 'test #', len(test_data))
train_data.head()
```

If you have a folder without `train` or `test` root folders, like:

- root/car/0001.jpg
- root/car/xxxa.jpg
- root/bus/123.png
- root/bus/023.jpg

Then you can load the splits with `from_folder`:

```{.python .input}
# use the train from shopee-iet as new root
root = os.path.dirname(train_data.iloc[0]['image'])
all_data = ImagePredictor.Dataset.from_folder(root)
all_data.head()
# you can manually split the dataset or use `random_split`
train, val, test = all_data.random_split(val_size=0.1, test_size=0.1)
print('train #', len(train), 'test #', len(test))
```

## Convert a list of images to dataset

You can create dataset from a list of images with a function, the function is used to determine the label of each image. We use the Oxford-IIIT Pet Dataset mini pack as an example, where images are scattered in `images` directory but with unique pattern: filenames of cat starts with capital letter, otherwise dogs. So we can use a function to distinguish and assign label to each image:

```{.python .input}
pets = ag.utils.download('https://autogluon.s3-us-west-2.amazonaws.com/datasets/oxford-iiit-pet-mini.zip')
pets = ag.utils.unzip(pets)
image_list = [x for x in os.listdir(os.path.join(pets, 'images')) if x.endswith('jpg')]
def label_fn(x):
    return 'cat' if os.path.basename(x)[0].isupper() else 'dog'
new_data = ImagePredictor.Dataset.from_name_func(image_list, label_fn, root=os.path.join(os.getcwd(), pets, 'images'))
new_data
```

## Visualize images

You can use `show_images` to visualize the images, as well as the corresponding labels:

```{.python .input}
new_data.show_images()
```

For raw DataFrame objects, you can convert them to Dataset first to use `show_images`.

Congratulations, you can now proceed to :ref:`sec_imgquick` to start training the `ImagePredictor`.
