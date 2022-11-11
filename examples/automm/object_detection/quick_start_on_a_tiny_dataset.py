import os
import time

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor


def tutorial_script_for_quick_start():
    zip_file = "s3://automl-mm-bench/object_detection_dataset/tiny_motorbike_coco.zip"
    download_dir = "./tiny_motorbike_coco"

    load_zip.unzip(zip_file, unzip_dir=download_dir)
    data_dir = os.path.join(download_dir, "tiny_motorbike")
    train_path = os.path.join(data_dir, "Annotations", "coco_trainval.json")
    test_path = os.path.join(data_dir, "Annotations", "coco_test.json")

    checkpoint_name = "yolov3_mobilenetv2_320_300e_coco"
    num_gpus = -1

    # Init predictor
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
        path="./quick_start_tutorial_temp_save",
    )

    start = time.time()

    # Fit
    predictor.fit(
        train_path,
        hyperparameters={
            "optimization.learning_rate": 2e-4, # we use two stage and detection head has 100x lr
            "optimization.max_epochs": 15,
            "env.per_gpu_batch_size": 32,  # decrease it when model is large
        },
    )

    train_end = time.time()
    print("The finetuning takes %.2f seconds." % (train_end - start))

    # Evaluate
    predictor.evaluate(test_path)

    eval_end = time.time()
    print("The evaluation takes %.2f seconds." % (eval_end - train_end))

    # Load and reset num_gpus
    new_predictor = MultiModalPredictor.load("./quick_start_tutorial_temp_save")
    new_predictor.set_num_gpus(1)

    # Evaluate new predictor
    new_predictor.evaluate(test_path)

if __name__ == "__main__":
    tutorial_script_for_quick_start()
