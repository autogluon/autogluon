# AutoMM for Named Entity Recognition

Named entity recognition (NER) refers to identifying and categorizing key information (entities) from unstructured text. An entity can be a word or a series of words which correspond to categories such as cities, time expressions, monetary values, facilities, person, organization, etc. An NER model usually takes as input an unannotated block of text and output an annotated block of text that highlights the named entities with predefined categories. For example, given the following sentences, 

- Albert Einstein was born in Germany and is widely acknowledged to be one of the greatest physicists.

The model will tell you that "Albert Einstein" is a PERSON and "Germany" is a LOCATION. In the following, we will introduce how to use AutoMM for the NER task, including how to prepare your data, how to train your model, and what you can expect from the model outputs.

 
## Prepare Your Data
Like other tasks in AutoMM, all you need to do is to prepare your data as data tables (i.e., dataframes) which contain a text column and an annotation column. The text column stores the raw textual data which contains the entities you want to identify. Correspondingly, the annotation column stores the label information (e.g., the *category* and the *start/end* offset in character level) for the entities. AutoMM requires the *annotation column* to have the following json format (Note: do not forget to call json.dumps() to convert python objects into a json string before creating your dataframe). 

- [{"entity_group": "PERSON", "start": 0, "end": 15}, 
{"entity_group": "LOCATION", "start": 28, "end": 35}]

where **entity_group** is the category of the entity and **start** is a character position indicating where the entity begins while **end** represents the ending position of the enity. To make sure that AutoMM can recognise your json annotations, it is required to use the exactly same keys/properties (entity_group, start, end) specified above when constructing your data. You can annote "Albert Einstein" as a single entity group or you can also assign each word a label.

If you are already familar with the NER task, you probably have heard about the [BIO](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) (Benginning-Inside-Outside) format. You can adopt this format (which is not compulsory) to add an *I-prefix* or a *B-prefix* to each tag to indicate wether the tag is the beginning of the annotated chunk or inside the chunk. For example, you can annotate "Albert" as "B-PERSON" because it is the beginning of the name and "Einstein" as "I-PERSON" as it is inside the PERSON chunk. You do not need to worry about the *O* tags, an *O* tag indicates that a word belongs to no chunk, as AutoMM will take care of that automatically. 

Now, let's look at an example dataset. This dataset is converted from the [MIT movies corpus](https://groups.csail.mit.edu/sls/downloads/movie/) which provides annotations on entity groups such as actor, character, director, genre, song, title, trailer, year, etc. 

```{.python .input}
from autogluon.core.utils.loaders import load_pd
train_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/mit-movies/train.csv')
test_data = load_pd.load('https://automl-mm-bench.s3.amazonaws.com/ner/mit-movies/test.csv')
train_data.head(5)
```

Let's print the first row.

```{.python .input}
print(f"text_snippet: {train_data['text_snippet'][0]}")
print(f"entity_annotations: {train_data['entity_annotations'][0]}")
```

## Training
Now, let's create a predictor for named entity recognition by seting the *problem_type* to **ner** and specifying the label column. Then we call predictor.fit() to train the model for five minutes. To achieve reasonable performance in your applications, you are recommended to set a longer enough time_limit (e.g., 30/60 minutes). You can also specify your backbone model and other hyperparameters using the hyperparameters argument. Here, we save the model to the path "./automm_ner".

```{.python .input}
from autogluon.multimodal import MultiModalPredictor
label_col = "entity_annotations"
predictor = MultiModalPredictor(problem_type="ner", label=label_col, path='./automm_ner')
predictor.fit(
    train_data=train_data,
    time_limit=300, #second
)
```

## Evaluation 
Evaluation is also straightforward, we use [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval) for NER evaluation and the supported metrics are *overall_recall*, *overall_precision*, *overall_f1*, *overall_accuracy*. If you are interested in seeing the performance on a specific entity group, you can use the entity group name as the evaluation metric with which you will obtain the performances (precision, recall, f1) on the given entity group:

```{.python .input}
predictor.evaluate(test_data,  metrics=['overall_recall', "overall_precision", "overall_f1", "actor"])
```

## Prediction 
You can easily obtain the predictions given an input sentence by by calling predictor.predict().

```{.python .input}
sentence = "Game of Thrones is an American fantasy drama television series created by David Benioff"
predictions = predictor.predict({'text_snippet': [sentence]})
print('Predicted entities:', predictions[0])

for entity in predictions[0]:
    print(f"Word '{sentence[entity['start']:entity['end']]}' belongs to group: {entity['entity_group']}")
```

## Reloading and Continuous Training 
The trained predictor is automatically saved and you can easily reload it using the path. If you are not saftisfied with the current model performance, you can continue training the loaded model with new data.

```{.python .input}
new_predictor = MultiModalPredictor.load('automm_ner')
new_predictor.fit(train_data, time_limit=60, save_path='automm_ner_continue_train')
test_score = new_predictor.evaluate(test_data, metrics=['overall_f1', 'ACTOR'])
print(test_score)
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.