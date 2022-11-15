from autogluon.multimodal import MultiModalPredictor
from ray import tune


def detection_hpo():
    train_data = "/media/code/detdata/DetBenchmark/clipart/Annotations/train_cocoformat.json"
    test_data = "/media/code/detdata/DetBenchmark/clipart/Annotations/test_cocoformat.json"
    num_trails = 8

    # simple HPO
    hyperparameters = {
        "optimization.learning_rate": tune.uniform(1e-4, 1e-2),
        "optimization.max_epochs": tune.choice(["10", "20"]),
        "model.names": ["mmdet_image"],
        "model.mmdet_image.checkpoint_name": tune.choice(["yolov3_mobilenetv2_320_300e_coco",
                                                         "faster_rcnn_r50_fpn_2x_coco"]),
    }

    hyperparameter_tune_kwargs = {
        "searcher": "bayes",  # random
        "scheduler": "ASHA",
        "num_trials": num_trails,
    }

    predictor = MultiModalPredictor(label="label",
                                    problem_type="object_detection",
                                    num_classes=6,
                                    classes=["a","b","c","d","e","f"],)

    predictor.fit(train_data=train_data,
                  hyperparameters=hyperparameters,
                  hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, )

    result = predictor.evaluate(test_data)
    print('Evaluation result:' % result)


if __name__ == "__main__":
    detection_hpo()
