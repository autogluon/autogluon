# How to obtain top 1% in Petfinder Pawpularity with AutoMM

Take [Petfinder Pawpularity competition](https://www.kaggle.com/competitions/petfinder-pawpularity-score/overview) as an example showing how to use MultiModalPredictor to complete a competition.

## 1. MultiModalPredictor for Training

MultiModalPredictor is a tool that can handle diverse data including images, text, numerical data and categorical data. It can automatically identify data types and fuse the pretrained DL backbones. With the tool, you can make the train easily with less code. For details, please refer to [`kaggle_Pawpularity.py`](./kaggle_Pawpularity.py).

### 1.1 Build the MultiModalPredictor

You can build the predictor as follow.

    predictor = MultiModalPredictor(
	    label="Pawpularity", 
	    problem_type="regression", 
	    eval_metric="rmse", 
	    path=save_path,  
	    verbosity=4, 
	)

 - `label` indicates the target value in training data.
 - `problem_type` indicates the type of the problem. That can be "Multiclass", "Binary" or "Regression".
 - `eval_metric` indicates the evaluation index of the model which is always the evaluation of the competition.
 - `path` indicates the path to save MultiModalPredictor models.
 - `verbosity` controls how much information is printed.

### 1.2 Train the MultiModalPredictor

Then, you can train the MultiModalPredictor with `.fit()`.

    predictor.fit(  
	    train_data=training_df,
	    tuning_data=valid_df,  
	    save_path=save_path,  
	    hyperparameters={
		    "model.names": "['timm_image']",
		    "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224",
		    "model.timm_image.train_transforms": "['resize_shorter_side','center_crop','randaug']",
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

 - `train_data` is the data used for training.
 - `tuning_data` is the data for validation. If it is empty, the tuning data will be split from training data automatically.
 - `save_path` indicates the specific path for model saving in a fit process.
 - `hyperparameters` is a Dict which will override the default configs in the training. The configs contain five different types.  
 -- `model` contains the parameters which control the models used in the predictor. You can select the model you need and adjust the details. Default is selecting the models determined by the dataset automatically.
 --`data` contains the configs of transforms for different types of data. 
 --`env` contains the configs of the training environment.
 --`optimization` contains the configs in the optimization process, including but not limited to max training epochs, learning rate and warm-up.
 - `seed` determines the random seed.

### 1.3 Save Standalone Model
In MultiModalPredictor, some pre-trained models will be downloaded during training. These models also need to be saved for use in predicting after submission. You can specify the predictor to save a “standalone” model that can be loaded without internet access.

    predictor.save(path=save_standalone_path, standalone=True)

## 2. Kaggle Kernel-only Competition with AutoGluon

In a Kaggle competition, especially a code competition, users cannot obtain AutoGluon resources through the network. 
To solve the problem, there are two key points:

 - Loading AutoGluon and its related libraries through datasets.
 - Using standalone models to avoid model downloading.

You can download [AutoGluon](https://github.com/autogluon/autogluon) and use the tools to train your model locally.
For using AutoGluon in Kaggle submission, it should be uploaded to Kaggle as a dataset. You can create a new dataset called "auotgluon" in Kaggle. After that, find AutoGluon at the installation path and upload it into the dataset. 
In this way, AutoGluon is introduced without network support in submission. 
Through the code following, you can import AutoGluon into your work.

    import sys
    sys.path.append("../input/autogluon/")
  
It should be noted that AutoGluon itself needs some dependencies which are not supported in the Kaggle environment. They should be introduced the same way as Auotgluon. 
Currently, these libraries need to be imported manually:

 - typish
 - timm
 - omegaconf
 - antlr4
 - nlpaug

You can refer to [Kaggle notebook](https://www.kaggle.com/code/linuxdex/get-autogluon-standalone) and run the notebook to get autogluon.multimodal standalone. Missing dependent packages will be downloaded as .whl or .tar.gz in autogluon_standalone. You can download the zip of the folder and create this as a Dataset in Kaggle.

Use the code following to install AutoGluon without network in the kaggle notebook. 

    import sys
    sys.path.append('../input/autogluon-standalone-install/autogluon_standalone/antlr4-python3-runtime-4.8/antlr4-python3-runtime-4.8/src/')
    !pip install --no-deps --no-index --quiet ../input/autogluon-standalone-install/autogluon_standalone/*.whl --find-links autogluon_standalone

Using the saved standalone model can avoid downloading models in submission. You can refer to *1.3* to save the standalone model.

## 3. Prediction in Kaggle Competitions

Next, let's upload the predictor to Kaggle and use it to generate predictions on the test set. You should upload the MultiModalPredictor standalone save files as a dataset to Kaggle and put it in the input data sources of your prediction code. 
You can load the MultiModalPredictor using the code following.

    pretrained_model = predictor.load(path=save_standalone_path)

With the `.predict()`, you can get the prediction of test datasets.

    test_pred = pretrained_model.predict(test_df)
 
For detailed codes, please refer to [`Kaggle_Pawpularity_submit.py`](./kaggle_Pawpularity_submit.py).

## 4. Result in Kaggle Competitions

You can refer to the [kaggle notebook](https://www.kaggle.com/code/linuxdex/use-autogluon-to-predict-pet-adoption) for submission and results.

| ID | Backbone | Fusion | Augment | learning_rate | lr_decay | weight_decay | Max_epochs | Warmup_step | Per_gpu_batch_size | Per_gpu_batch_size_evaluation | Precision | CV | Public_Leaderboard | Private_Leaderboard | Download |
|----|----------|--------|---------|---------------|----------|--------------|------------|-------------|--------------------|-------------------------------|-----------|----|--------------------|---------------------|----------|
| 1 | vit_large_patch16_384 | True | randaug | 2e-5 | 1 | 0 | 5 | 0 | 8 | 3 | 32 | 17.3740541397068 | 17.97642 | 17.10867| [result7](http://automl-mm-bench.s3.amazonaws.com/0.5release/petfinder_pawpularity/result7_standalone.zip) |
| 2 | swin_large_patch4_window12_384 | True | randaug | 2e-5 | 1 | 0 | 5 | 0 | 8 | 32 | 32 | 17.4974990118305 | 18.09335 | 17.18875 | [result6](http://automl-mm-bench.s3.amazonaws.com/0.5release/petfinder_pawpularity/result6_standalone.zip) |
| 3 | convnext_large_384_in22ft2k | False | randaug | 5e-5 | 1 | 0 | 10 | 0 | 8 | 4 | 32 | 17.4523797944187 | 18.25999 | 17.20016 | [result26](http://automl-mm-bench.s3.amazonaws.com/0.5release/petfinder_pawpularity/result26_standalone.zip) |
| 4 | swin_large_patch4_window7_224 | False | randaug | 5e-5 | 1 | 0 | 5 | 0 | 16 | 32 | 32 | 17.5192244849318 | 18.03887 | 17.27713 | [result13](http://automl-mm-bench.s3.amazonaws.com/0.5release/petfinder_pawpularity/result13_standalone.zip) |
| 5 | swin_large_patch4_window7_224 | True | randaug | 5e-5 | 1 | 0 | 10 | 0.1 | 16 | 32 | 32 | 17.4848481619876 | 18.15082 | 17.29325 | [result30](http://automl-mm-bench.s3.amazonaws.com/0.5release/petfinder_pawpularity/result30_standalone.zip) |
| 6 | vit_large_patch16_384 | False | randaug | 2e-5 | 1 | 0 | 5 | 0 | 8 | 3 | 32 | 17.5162709909151 | 18.15326 | 17.37978 | [result23](http://automl-mm-bench.s3.amazonaws.com/0.5release/petfinder_pawpularity/result23_standalone.zip) |

The results of the model are shown in the table above.

It is obvious that the model using fusion usually has better results.

| BackBone | Fusion_CV | Image_only_CV |
|----------|-----------|---------------|
| swin_large_patch4_window7_224 | 17.4848481619876 | 17.5192244849318 |
| swin_large_patch4_window12_384 | 17.4974990118305 | 17.5871592343891 |
| convnext_large_384_in22ft1k | 17.6218877694844 | 17.4523797944187 |
| vit_large_patch16_384 | 17.3740541397068 | 17.5162709909151 | 
| beit_large_patch16_384 | 17.530005178868 | 17.6423355406175 |

With MultiModalPredictor, You can easily use fusion through modifying hyper parameters instead of changing codes.

The results of model ensemble are as follows.

| Ensemble IDs | Weights | Public_Leaderboard | Private_Leaderboard | Kaggle screenshot |
|--------------|---------|--------------------|---------------------|-------------------|
| [1, 2, 3] | [0.5, 0.25, 0.25] | 17.91944 | 16.97737 | [Kaggle result](http://automl-mm-bench.s3.amazonaws.com/0.5release/petfinder_pawpularity/kaggle-shot/shot_1.png) |
| [1, 2, 3, 4] | [0.25, 0.25, 0.25, 0.25] | 17.90128 | 16.97970 | [Kaggle result](http://automl-mm-bench.s3.amazonaws.com/0.5release/petfinder_pawpularity/kaggle-shot/shot_2.png) |
| [1, 2, 4] | [0.4, 0.4, 0.2] | 17.88050 | 16.99416 | [Kaggle result](http://automl-mm-bench.s3.amazonaws.com/0.5release/petfinder_pawpularity/kaggle-shot/shot_3.png) |
| [1, 2] | [0.5, 0.5] | 17.91753 | 17.01075 | [Kaggle result](http://automl-mm-bench.s3.amazonaws.com/0.5release/petfinder_pawpularity/kaggle-shot/shot_4.png) |
