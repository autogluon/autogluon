from autogluon.vision import ImagePredictor, ImageDataset
import autogluon.core as ag
import os
import pandas as pd
import numpy as np
import copy


def test_task():
    dataset, _, test_dataset = ImageDataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    model_list = ImagePredictor.list_models()
    classifier = ImagePredictor()
    classifier.fit(dataset, hyperparameters={'epochs': 1, 'early_stop_patience': 3}, hyperparameter_tune_kwargs={'num_trials': 2})
    assert classifier.fit_summary()['valid_acc'] > 0.1, 'valid_acc is abnormal'
    test_result = classifier.predict(test_dataset)
    single_test = classifier.predict(test_dataset.iloc[0]['image'])
    single_proba = classifier.predict_proba(test_dataset.iloc[0]['image'])
    classifier.save('classifier.ag')
    classifier2 = ImagePredictor.load('classifier.ag')
    fit_summary = classifier2.fit_summary()
    test_acc = classifier2.evaluate(test_dataset)
    # raw dataframe
    df_test_dataset = pd.DataFrame(test_dataset)
    test_acc = classifier2.evaluate(df_test_dataset)
    assert test_acc['top1'] > 0.2, f'{test_acc} too bad'
    test_proba = classifier2.predict_proba(test_dataset)
    test_feature = classifier2.predict_feature(test_dataset)
    single_test2 = classifier2.predict(test_dataset.iloc[0]['image'])
    assert isinstance(single_test2, pd.Series)
    assert single_test2.equals(single_test)
    # to numpy
    test_proba_numpy = classifier2.predict_proba(test_dataset, as_pandas=False)
    assert np.array_equal(test_proba.to_numpy(), test_proba_numpy)
    test_feature_numpy = classifier2.predict_feature(test_dataset, as_pandas=False)
    single_test2_numpy = classifier2.predict(test_dataset.iloc[0]['image'], as_pandas=False)
    assert np.array_equal(single_test2.to_numpy(), single_test2_numpy)


def test_task_label_remap():
    train_dataset, _, test_dataset = ImageDataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    label_remap = {0: 'd', 1: 'c', 2: 'b', 3: 'a'}
    train_dataset = train_dataset.replace({"label": label_remap})
    test_dataset = test_dataset.replace({"label": label_remap})
    # rename label column
    train_dataset = train_dataset.rename(columns={'label': 'my_label'})
    predictor = ImagePredictor(label='my_label')
    dataset_copy = copy.deepcopy(train_dataset)
    predictor.fit(train_dataset, hyperparameters={'epochs': 2})
    # assert input dataset not altered
    assert dataset_copy.equals(train_dataset)
    pred = predictor.predict(test_dataset)
    pred_proba = predictor.predict_proba(test_dataset)
    label_remap_inverse = {col_name: i for i, col_name in enumerate(list(pred_proba.columns))}
    from autogluon.core.metrics import accuracy, log_loss
    score_accuracy = accuracy(y_true=test_dataset['label'], y_pred=pred)
    score_log_loss = log_loss(y_true=test_dataset['label'].replace(label_remap_inverse), y_pred=pred_proba.to_numpy())
    assert score_accuracy > 0.2  # relax


def test_invalid_image_dataset():
    invalid_test = ag.download('https://autogluon.s3-us-west-2.amazonaws.com/miscs/test_autogluon_invalid_dataset.zip')
    invalid_test = ag.unzip(invalid_test)
    df = ImageDataset.from_csv(os.path.join(invalid_test, 'train.csv'), root=os.path.join(invalid_test, 'train_images'))
    predictor = ImagePredictor(label="labels")
    predictor.fit(df, df.copy(), time_limit=60)


def test_image_predictor_presets():
    train_dataset, _, test_dataset = ImageDataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    for preset in ['medium_quality_faster_train',]:
        predictor = ImagePredictor()
        predictor.fit(train_dataset, tuning_data=test_dataset, presets=[preset], time_limit=300, hyperparameters={'epochs': 1})
