Take [Petfinder Pawpularity competition](https://www.kaggle.com/competitions/petfinder-pawpularity-score/overview) as an example showing how to use AutoMMPredictor to complete a competition.

## 1. Kaggle Kernel-only Competition with AutoGluon

In a kaggle competition, especially a code competition, users cannot obtain AutoGluon resources through the network. 
To solve the problem, there are two key points:

 - Get AutoGluon and its dependent libraries.
 - Save the downloaded resources when saving your model.

### 1.1 Get AluoGluon in Kaggle

You can download [AutoGluon](https://github.com/awslabs/autogluon) and use the tools to train your own model locally.
For using AutoGluon in kaggle submission, it should be uploaded to kaggle as a dataset. You can create a new dataset called "auotgluon" in kaggle. After that, finding AutoGluon at the installation path and upload it into the dataset. 
In this way, AutoGluon is introduced without network support in submission. 
Through the code follow, you can import AutoGluon in your work.

    import sys
    sys.path.append("../input/autogluon/")
  
 It should be noted that AutoGluon itself needs some dependencies which are not supported in the kaggle environment. They should be introduced the same way as Auotgluon. 
Currently, these libraries need to be imported manually:

 - nptyping
 - typish
 - timm
 - omegaconf
 - antlr4

### 1.2 Save Standalone Model
In AutoMMPredictor, some pretrained models will be downloaded during training. These models also need to be saved for use in predicting after submission. You can open standalone in model saving to save the downloaded models.

    predictor.save(path=save_standalone_path,standalone=True)

## 2. AutoMMPredictor for Training

AutoMMPredictor is a tool that can handle diverse data including image, text, numerical data and categorical data. It can automatically identify data types and perform fusion. With the tool, you can make the train easily with less code. For details, please refer to [`kaggle_Pawpularity.py`](./kaggle_Pawpularity.py).

### 2.1 Build the AutoMMPredictor

You can build the predictor as follow.

    predictor = AutoMMPredictor(
	    label="Pawpularity", 
	    problem_type="regression", 
	    eval_metric="rmse", 
	    path=save_path,  
	    verbosity=4, 
	)

 - `label`indicates the target value in training data.
 - `problem_type`indicates the type of the problem. That can be "Multiclass", "Binary" or "Regression".
 - `eval_metric`indicates the evaluation index of the model which is always the evaluation of the competition.
 - `path`indicates the path to save AutoMMPredictor models.
 - `verbosity`controls how much information is printed.

### 2.2 Train the AutoMMPredictor

 Then, you can train the AutoMMPredictor with `.fit()`.

    predictor.fit(  
	    train_data=training_df,
	    tuning_data=valid_df,  
	    save_path=save_path,  
	    hyperparameters={
		    "model.names": "['timm_image']",
		    "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224",
		    "model.timm_image.train_transform_types": "['resize_shorter_side','center_crop','randaug']",
		    "data.categorical.convert_to_text": "False",
		    "env.per_gpu_batch_size": "16",
		    "env.per_gpu_batch_size_evaluation": "32",
		    "env.precision": "32",
		    "optimization.learning_rate": "2e-5",
		    "optimization.weight_decay": "0",
		    "optimization.lr_decay": "1",
		    "optimization.max_epochs": "5",
		    "optimization.warmup_steps": "0",
		    "optimization.loss_function": "bcewithlogitsloss",
		},
		seed=1,
	)

 - `train_data`is the data used for training.
 - `tuning_data`is the data for validation. If it is empty, the tuning data will be split from training data automatically.
 - `save_path`indicates the specific path for model saving in a fit process.
 - `hyperparameters`is a Dict which will override the default configs in the training. The configs contain five different types.  
 -- `model`contains the parameters which control the models used in the predictor. You can select the model you need and adjust the details. Default is selecting the models determined by dataset automatically.
 --`data`contains the configs of transforms for different types of data. 
 --`env`contains the configs of the training environment.
 --`optimization`contains the configs in the optimization process, including but not limit to max training epochs, learning rate and warm up.
 - `seed`determines the random seed.

## 3. Prediction in Kaggle Competitions

In kaggle competitions, the predictor you trained will be used to predict the test dataset. You should upload the AutoMMPredictor standalone save files as a dataset to kaggle and put it in the input data sources of your prediction code. 
You can load the AutoMMPredictor using the code follow.

    pretrained_model = predictor.load(path=save_standalone_path)

With the `.predict()`, you can get the prediction of test datasets.

    test_pred = pretrained_model.predict(test_df)
 
 For detail codes, please refer to [`kaggle_Pawpularity_submit.py`](./kaggle_Pawpularity_submit.py).
