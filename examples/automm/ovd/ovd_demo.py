from autogluon.multimodal import MultiModalPredictor


def ovd():

    # Train
    predictor = MultiModalPredictor(problem_type="open_vocabulary_object_detection")
    predictor.set_num_gpus(1)

    # Evaluation on test dataset
    test_result = predictor.predict(
        {
            "images": [
                "/home/haoyfang/datasets/AGDetBench/cityscapes/images/leftImg8bit/train/zurich/zurich_000006_000019_leftImg8bit.png",
                "/home/haoyfang/datasets/AGDetBench/cityscapes/images/leftImg8bit/train/zurich/zurich_000005_000019_leftImg8bit.png",
            ],
            "prompts": [
                "car. bicycle. traffic light. people. building. tree.",
                "car. bicycle. people. building. tree.",
            ],
        }
    )
    print("test_result:")
    print(test_result)


if __name__ == "__main__":
    ovd()
