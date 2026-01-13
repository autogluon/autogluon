from autogluon.multimodal import MultiModalPredictor


def image_regression():
    # Prepare data
    from autogluon.multimodal.utils.misc import shopee_dataset

    download_dir = "./ag_automm_tutorial_imgcls"
    train_data_path, test_data_path = shopee_dataset(download_dir)
    print(train_data_path)

    # Train
    predictor = MultiModalPredictor(problem_type="regression", label="label")
    predictor.fit(train_data=train_data_path, hyperparameters={"optim.max_epochs": 3, "env.batch_size": 8})

    # Evaluation on test dataset
    test_result = predictor.predict(test_data_path)
    print("test_result:")
    print(test_result)

    test_evaluation = predictor.evaluate(test_data_path)
    print("Evaluation result:", test_evaluation)

    result = predictor.predict(test_data_path.iloc[0]["image"])
    print("Prediction result for single image:")
    print(result)


if __name__ == "__main__":
    image_regression()
