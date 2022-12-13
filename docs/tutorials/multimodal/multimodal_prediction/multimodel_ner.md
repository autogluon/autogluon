# AutoMM for Multimodal Named Entity Extraction

We have introduced how to train an entity extraction model with text data.
Here, we move a step further by integrating data of other modalities.
In many real-world applications, textual data usually comes with data of other modalities.
For example, Twitter allows you to compose tweets with text, photos, videos, and GIFs. Amazon.com uses text, images, and videos to describe their products.
These auxiliary modalities can be leveraged as additional context resolution of entities.
Now, with AutoMM, you can easily exploit multimodal data to enhance entity extraction without worrying about the details.

```{.python .input}
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
```

## Get the Twitter Dataset
In the following example, we will demonstrate how to build a multimodal named entity recognition model with a real-world [Twitter dataset](https://github.com/jefferyYu/UMT/tree/master).
This dataset consists of scrapped tweets from 2016 to 2017, and each tweet was composed of one sentence and one image. Let's download the dataset.

```{.python .input}
download_dir = './ag_automm_tutorial'
zip_file = 'https://automl-mm-bench.s3.amazonaws.com/ner/multimodal_ner.zip'
from autogluon.core.utils.loaders import load_zip
load_zip.unzip(zip_file, unzip_dir=download_dir)
```

Next, we will load the CSV files.

```{.python .input}
dataset_path = download_dir + '/multimodal_ner'
train_data = pd.read_csv(f'{dataset_path}/twitter17_train.csv')
test_data = pd.read_csv(f'{dataset_path}/twitter17_test.csv')
label_col = 'entity_annotations'
```

We need to expand the image paths to load them in training.

```{.python .input}
image_col = 'image'
train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0]) # Use the first image for a quick tutorial
test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

def path_expander(path, base_folder):
	path_l = path.split(';')
	p = ';'.join([os.path.abspath(base_folder+path) for path in path_l])
	return p

train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

train_data[image_col].iloc[0]
```

Each row consists of the text and image of a single tweet and the entity_annotataions which contains the named entity annotations for the text column.
Let’s look at an example row and display the text and picture of the tweet.

```{.python .input}
example_row = train_data.iloc[0]

example_row
```

Below is the image of this tweet.

```{.python .input}
example_image = example_row[image_col]

from IPython.display import Image, display
pil_img = Image(filename=example_image, width =300)
display(pil_img)
```

As you can see, this photo contains the logos of the Real Madrid football club, Manchester United football club, and the UEFA super cup. Clearly, the key information of the tweet sentence is coded here in a different modality.

## Training
Now let’s fit the predictor with the training data.
Firstly, we need to specify the problem_type to **ner**.
And to ensure the model to locate the correct text column for entity extraction, we need to set the corresponding column type to `ner` using the **column_types** parameter.
Here we set a tight time budget for a quick demo.

```{.python .input}
from autogluon.multimodal import MultiModalPredictor
import uuid

label_col = "entity_annotations"
model_path = f"./tmp/{uuid.uuid4().hex}-automm_multimodal_ner"
predictor = MultiModalPredictor(problem_type="ner", label=label_col, path=model_path)
predictor.fit(
	train_data=train_data,
	column_types={"text_snippet":"ner"},
	time_limit=300, #second
)
```

Under the hood, AutoMM automatically detects the data modalities, selects the related models from the multimodal model pools, and trains the selected models.
If multiple backbones are available, AutoMM appends a late-fusion model on top of them.

## Evaluation

```{.python .input}
predictor.evaluate(test_data,  metrics=['overall_recall', "overall_precision", "overall_f1"])
```

## Prediction

You can easily obtain the predictions by calling predictor.predict().

```{.python .input}
prediction_input = test_data.drop(columns=label_col).head(1)
predictions = predictor.predict(prediction_input)
print('Tweet:', prediction_input.text_snippet[0])
print('Image path:', prediction_input.image[0])
print('Predicted entities:', predictions[0])

for entity in predictions[0]:
	print(f"Word '{prediction_input.text_snippet[0][entity['start']:entity['end']]}' belongs to group: {entity['entity_group']}")
```

## Reloading and Continuous Training

The trained predictor is automatically saved and you can easily reload it using the path.
If you are not satisfied with the current model performance, you can continue training the loaded model with new data.

```{.python .input}
new_predictor = MultiModalPredictor.load(model_path)
new_model_path = f"./tmp/{uuid.uuid4().hex}-automm_multimodal_ner_continue_train"
new_predictor.fit(train_data, time_limit=60, save_path=new_model_path)
test_score = new_predictor.evaluate(test_data, metrics=['overall_f1'])
print(test_score)
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
