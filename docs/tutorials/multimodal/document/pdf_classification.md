# Classifying PDF Documents with AutoMM.
PDF comes short from Portable Document Format and is one of the most popular document formats.
We can find PDFs everywhere, from personal resumes to business contracts, and from commercial brochures to government documents. 
The list can be endless. 
PDF is highly praised for its portability. 
There's no worry about the receiver being unable to view the document or see an imperfect version regardless of their operating system and device models.

Using AutoMM, you can handle and build machine learning models on PDF documents just like working on other modalities such as text and images, without bothering about PDFs processing. 
In this tutorial, we will introduce how to sort PDF documents automatically with AutoMM using document foundation models. Let’s get started!


## Get the PDF document dataset
We have created a simple PDFs dataset via manual crawling for demonstration purpose. 
It consists of two categories, resume and historical (downloaded from [milestone documents](https://www.archives.gov/milestone-documents/list)). 
We picked 20 PDF documents for each of the category. 

Now, let's download the dataset and split it into training and test sets.

```{.python .input}
import os
import pandas as pd
from autogluon.core.utils.loaders import load_zip

download_dir = './ag_automm_tutorial_pdf_classifier'
zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/pdf_docs_small.zip"
load_zip.unzip(zip_file, unzip_dir=download_dir)

dataset_path = os.path.join(download_dir, "pdf_docs_small")
pdf_docs = pd.read_csv(f"{dataset_path}/data.csv")
train_data = pdf_docs.sample(frac=0.8, random_state=200)
test_data = pdf_docs.drop(train_data.index)
```

Now, let's visualize one of the PDF documents. Here, we use the S3 URL of the PDF document and `IFrame` to show it in the tutorial.
```{.python .input}
from IPython.display import IFrame

IFrame("https://automl-mm-bench.s3.amazonaws.com/doc_classification/historical_1.pdf", width=400, height=500)
```
As you can see, this document is an America's historical document in PDF format. 
To make sure the tutorial can locate the documents correctly, we need to overwrite the document paths.
```{.python .input}
from autogluon.multimodal.utils.misc import path_expander

DOC_PATH_COL = "doc_path"

train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
print(test_data.head())
```

## Create a PDF Document Classifier

You can create a PDFs classifier easily with `MultiModalPredictor`. 
All you need to do is to create a predictor and fit it with the above training dataset. 
AutoMM will handle all the details, like (1) detecting if it is PDF format datasets; (2) processing PDFs like converting it into a format that our model can recognize; (3) detecting and recognizing the text in PDF documents; etc., without your notice. 

Here, label is the name of the column that contains the target variable to predict, e.g., it is “label” in our example. 
We set the training time limit to 120 seconds for demonstration purposes.
```{.python .input}
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label")
predictor.fit(
    train_data=train_data,
    hyperparameters={"model.document_transformer.checkpoint_name":"microsoft/layoutlm-base-uncased",
    "optimization.top_k_average_method":"best",
    },
    time_limit=120,
)
```

## Evaluate on Test Dataset

You can evaluate the classifier on the test dataset to see how it performs:

```{.python .input}
scores = predictor.evaluate(test_data, metrics=["accuracy"])
print('The test acc: %.3f' % scores["accuracy"])
```

## Predict on a New PDF Document

Given an example PDF document, we can easily use the final model to predict the label:
```{.python .input}
predictions = predictor.predict({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(f"Ground-truth label: {test_data.iloc[1]['label']}, Prediction: {predictions}")
```

If probabilities of all categories are needed, you can call predict_proba:
```{.python .input}
proba = predictor.predict_proba({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(proba)
```

## Extract Embeddings

Extracting representation from the whole document learned by a model is also very useful. 
We provide extract_embedding function to allow predictor to return the N-dimensional document feature where N depends on the model.
```{.python .input}
feature = predictor.extract_embedding({DOC_PATH_COL: [test_data.iloc[0][DOC_PATH_COL]]})
print(feature[0].shape)
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
