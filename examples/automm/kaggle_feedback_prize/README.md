# Using MultiModalPredictor with Text Normalization

Take [Feedback Prize - Predicting Effective Arguments](https://www.kaggle.com/competitions/feedback-prize-effectiveness) as an example to show how text normalization can be helpful.

## 1. Preprocess data with text normalization

Text normalization is the task of mapping non-canonical language, typical of speech transcription and computer-mediated communication, to a standardized writing. 
It is an up-stream task necessary to enable the subsequent direct employment of standard natural language processing tools and indispensable for languages such as Swiss German, 
with strong regional variation and no written standard. 
Even though the competition dataset is composed of English only, we found that applying text normalization can reduce `log loss`, the metrics that this competition is evaluating on. 

### 1.1 Applying normalization to text columns by enabling hyperparameter configuration
    hyperparameters={
        "data.text.normalize_text": True,
    }

Under the hood, the following process is done when `data.text.normalize_text` is enabled:
    
    #### 1.1.1 Define error handlers for codecs
        def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
            return error.object[error.start : error.end].encode("utf-8"), error.end

        def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
        return error.object[error.start : error.end].decode("cp1252"), error.end


    #### 1.1.2 Register error handlers for codecs
        codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
        codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


    #### 1.1.3 Applying a series of decoding and encoding for normalization
        def resolve_encodings_and_normalize(text: str) -> str:
            text = (
                text.encode("raw_unicode_escape")
                .decode("utf-8", errors="replace_decoding_with_cp1252")
                .encode("cp1252", errors="replace_encoding_with_utf8")
                .decode("utf-8", errors="replace_decoding_with_cp1252")
            )
            text = unidecode(text)
            return text

### 1.2 A few examples of normalized texts

    # Example-1 pre-normalization
    'The same technology can make computer-animated faces more expressive\x97for video games or video surgery. \x93Most human communication is nonverbal, including emotional communication,\x94 notes Dr. Huang. \x93So computers need to understand that, too.\x94Eckman has classified six basic emotions\x97happiness, surprise, anger, disgust, fear, and sadness\x97and then associated each with characteristic movements of the facial muscles. For example, your frontalis pars lateralis muscle (above your eyes) raises your eyebrows when you\x92re surprised; your orbicularis oris (around your mouth) tightens your lips to show anger. '

    # Example-1 post-normalization
    'The same technology can make computer-animated faces more expressive--for video games or video surgery. "Most human communication is nonverbal, including emotional communication," notes Dr. Huang. "So computers need to understand that, too."Eckman has classified six basic emotions--happiness, surprise, anger, disgust, fear, and sadness--and then associated each with characteristic movements of the facial muscles. For example, your frontalis pars lateralis muscle (above your eyes) raises your eyebrows when you\'re surprised; your orbicularis oris (around your mouth) tightens your lips to show anger. '

    # Example-2 pre-normalization
    '"Congestion was down\xa0 60\xa0 percent\xa0 in\xa0 the\xa0 capital\xa0 of\xa0 france\xa0 after\xa0 fivedays\xa0 of\xa0 intensifying\xa0 smog."smog\xa0 by\xa0 meaning\xa0 pollution\xa0 went\xa0 down\xa0 just\xa0 60\xa0 percent\xa0 in\xa0 five\xa0 days. Thats\xa0 a\xa0 great\xa0 advantage\xa0 just\xa0 by\xa0 limiting\xa0 car\xa0 usage. In\xa0 source\xa0 number 1\xa0 explains that " Passenger\xa0 cars\xa0 are\xa0 responsible\xa0 for\xa0 12\xa0 percent\xa0 of\xa0 greenhouse\xa0 gas emissions\xa0 in\xa0 Europe.. and up\xa0 to\xa0 50\xa0 perecnt\xa0 in\xa0 some\xa0 car-intensive\xa0 areas\xa0 in\xa0 the\xa0 Untied States." We\xa0 as\xa0 a\xa0 country\xa0 should\xa0 lower\xa0 that\xa0 and\xa0 the\xa0 best\xa0 way\xa0 is\xa0 to\xa0 limiting\xa0 car usage . Limiting car usage is one of te some advantages to lowering pollution ( greenhouse gas, smog). '

    # Example-2 post-normalization
    '"Congestion was down  60  percent  in  the  capital  of  france  after  fivedays  of  intensifying  smog."smog  by  meaning  pollution  went  down  just  60  percent  in  five  days. Thats  a  great  advantage  just  by  limiting  car  usage. In  source  number 1  explains that " Passenger  cars  are  responsible  for  12  percent  of  greenhouse  gas emissions  in  Europe.. and up  to  50  percent  in  some  car-intensive  areas  in  the  Untied States." We  as  a  country  should  lower  that  and  the  best  way  is  to  limiting  car usage . Limiting car usage is one of te some advantages to lowering pollution ( greenhouse gas, smog). '

For details, please refer to 
[`kaggle_feedback_prize_preprocess.py`](./kaggle_feedback_prize_preprocess.py).

## 2. MultiModalPredictor for Training

MultiModalPredictor can automatically build deep learning models with multimodal datasets. 
The tabular data we have for this Kaggle competition is a perfect example that showcases how easily we could build models with just a few lines of code using MultiModalPredictor. 
For details, please refer to [`kaggle_feedback_prize_train.py`](./kaggle_feedback_prize_train.py).

### 2.1 Build the MultiModalPredictor

You can build the predictor as following.

    predictor = MultiModalPredictor(
	    label="discourse_effectiveness", 
	    problem_type="multiclass", 
	    eval_metric="log_loss", 
	    path=save_path,  
	    verbosity=3, 
	)

 - `label` indicates the target value in training data.
 - `problem_type` indicates the type of the problem. It can be "multiclass", "binary" or "regression".
 - `eval_metric` indicates the evaluation metrics of the model which is always the evaluation of the competition.
 - `path` indicates the path to save MultiModalPredictor models.
 - `verbosity` controls how much information is printed.

### 2.2 Train the MultiModalPredictor

Then, you can train the MultiModalPredictor with `.fit()`.

    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        presets="best_quality",
        hyperparameters={
            "model.hf_text.checkpoint_name": "microsoft/deberta-v3-large",
            "data.text.normalize_text": True,
            "optimization.learning_rate": 5e-5,
            "optimization.max_epochs": 7,
        },
    )

 - `train_data` is the data used for training.
 - `tuning_data` is the data for validation. If it is empty, the tuning data will be split from training data automatically.
 - `presets` sets a various number of parameters depending on the quality of models one prefers. For details, please refer to [presets section](https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-quickstart.html#presets)
 - `hyperparameters` is a Dict which will override the default configs in the training. The configs contain five different types.  
 -- `model` contains the parameters which control the models used in the predictor. You can select the model you need and adjust the details. Default is selecting the models determined by the dataset automatically.
 --`optimization` contains the configs in the optimization process, including but not limited to max training epochs, learning rate and warm-up.

### 2.3 Save Standalone Model
Models should be saved for offline deployment for Kaggle competitions, and uploaded to Kaggle as `datasets` after training is done. You can specify the MultiModalPredictor to save a “standalone” model that can be loaded without internet access.

    predictor.save(path=save_standalone_path, standalone=True)

## 2. Kaggle Kernel-only Competition with AutoGluon

In a Kaggle competition, especially a code competition, users cannot obtain AutoGluon resources through the network. 
To solve the problem, there are two key points:

 - Loading AutoGluon and its related libraries through datasets.
 - Using standalone models to avoid model downloading.

The AutoGluon and its dependencies are currently packaged in a zip file and available for downloading as data in [Kaggle notebook](https://www.kaggle.com/code/linuxdex/get-autogluon-standalone/data). 
You can download `autogluon_standalone.zip`, unzip it, and upload this folder as a [Kaggle Dataset](https://www.kaggle.com/datasets).

Use the following code to install AutoGluon without network in a kaggle notebook. 

    import sys
    sys.path.append("../input/autogluon-standalone/antlr4-python3-runtime-4.8/antlr4-python3-runtime-4.8/src/")
    !pip install --no-deps --no-index --quiet ../input/autogluon-standalone/autogluon_standalone/*.whl

Using the saved standalone model can avoid downloading models in submission. You can refer to [Section 2.3](#23-save-standalone-model) to save the standalone model.

## 3. Prediction in Kaggle Competitions

Next, let's upload the predictor to Kaggle and use it to generate probabilities on the test set. You can upload the MultiModalPredictor standalone models as datasets to Kaggle directly on a notebook, 
or via [Kaggle API](https://www.kaggle.com/docs/api#interacting-with-datasets). 
Make sure that models are present under the `Input` section in your notebook.

You can then load the MultiModalPredictor using the following code.

    pretrained_model = MultiModalPredictor.load(path=save_standalone_path)

You can upload the [preprocessing script](./kaggle_feedback_prize_preprocess.py) to Kaggle following the [instructions](https://www.kaggle.com/product-feedback/91185) or paste them directly into a notebook code block.

Preprocess test data with text normalization.

    test_df = kaggle_feedback_prize_preprocess.read_and_process_data_with_norm(data_path, "test.csv", is_train=False)

With the `.predict_proba()`, you can get the probabilities of all classes.

    test_pred = pretrained_model.predict_proba(test_df)
 
For detailed codes, please refer to [`kaggle_feedback_prize_submit.py`](./kaggle_feedback_prize_submit.py).

## 4. Benchmarking model performance with text normalization

We have benchmarked text normalization effect on different models and hyperparameters. 
For model evaluation, we fixed 20% of stratified samples from the training data, and we also submitted a couple of large models to Kaggle competition for leadboard scores. 


| model | lr | lr_decay | cv_k | normalized_text | local_log_loss  | kaggle_private | kaggle_public 
| --- | --- | --- | --- |--- |--- |--- |---
| microsoft/deberta-v3-base | 5e-5 | 0.9 | 3 | Y | 0.5692
| microsoft/deberta-v3-base | 5e-5 | 0.9 | 3 | N | 0.5835
| microsoft/deberta-v3-base | 5e-5 | 0.9 | 5 | Y | 0.5694
| microsoft/deberta-v3-base | 5e-5 | 0.9 | 5 | N | 0.5750
| microsoft/deberta-v3-large | 5e-5 | 0.9 | 3 | Y | 0.5848
| microsoft/deberta-v3-large | 5e-5 | 0.9 | 3 | N | 0.5779
| microsoft/deberta-v3-large | 5e-5 | 0.9 | 5 | Y | 0.5552 | 0.621 | 0.6267
| microsoft/deberta-v3-large | 5e-5 | 0.9 | 5 | N | 0.5703 | 0.6228 | 0.6296
| roberta-base | 5e-5 | 0.9 | 3 | Y | 0.5969
| roberta-base | 5e-5 | 0.9 | 3 | N | 0.5944
| roberta-base | 5e-5 | 0.9 | 5 | Y | 0.5741
| roberta-base | 5e-5 | 0.9 | 5 | N | 0.5781
| roberta-large | 5e-5 | 0.9 | 3 | Y | 0.5739
| roberta-large | 5e-5 | 0.9 | 3 | N | 0.5850
| roberta-large | 5e-5 | 0.9 | 5 | Y | 0.5635 | 0.6419 | 0.6399
| roberta-large | 5e-5 | 0.9 | 5 | N | 0.5657 | 0.6439 | 0.6404

The results of the benchmark are shown in the table above. It is evident that text normalization is effective in majority of the cases.
