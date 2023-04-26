from autogluon.multimodal import MultiModalPredictor, download


def ovd():
    sample_image_path = download("https://live.staticflickr.com/65535/49004630088_d15a9be500_6k.jpg")

    # Train
    predictor = MultiModalPredictor(problem_type="open_vocabulary_object_detection")

    # Evaluation on test dataset
    test_result = predictor.predict(
        {
            "image": [sample_image_path],
            "prompt": [
                "Pink notice. Green sign. One Way sign. People group. Tower crane in construction. Lamp post. Glass skyscraper."
            ],
        },
        as_pandas=True,
    )
    print("test_result:")
    print(test_result)


if __name__ == "__main__":
    ovd()
