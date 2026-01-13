import json
import os
import time

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor


def tutorial_script_for_quick_start():
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/tiny_motorbike_coco.zip"
    download_dir = "./tiny_motorbike_coco"

    load_zip.unzip(zip_file, unzip_dir=download_dir)
    data_dir = os.path.join(download_dir, "tiny_motorbike")
    train_path = os.path.join(data_dir, "Annotations", "trainval_cocoformat.json")
    test_path = os.path.join(data_dir, "Annotations", "test_cocoformat.json")
    test_path_no_label = os.path.join(data_dir, "Annotations", "test_cocoformat_no_label.json")

    with open(test_path, "r") as f:
        test_data = json.load(f)

    test_data.pop("annotations")
    test_data.pop("categories")
    print(test_data.keys())

    with open(test_path_no_label, "w+") as f_nolabel:
        json.dump(test_data, f_nolabel)

    checkpoint_name = "yolov3_mobilenetv2_8xb24-320-300e_coco"
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
            "optim.lr": 2e-4,  # we use two stage and detection head has 100x lr
            "optim.max_epochs": 20,
            "env.per_gpu_batch_size": 32,  # decrease it when model is large
        },
    )

    train_end = time.time()
    print("The finetuning takes %.2f seconds." % (train_end - start))

    # Evaluate
    # print("Predict...")
    # predictor.predict(test_path)

    # sample_image_path = "/home/ubuntu/ag/autogluon/examples/automm/object_detection/tiny_motorbike_coco/tiny_motorbike/JPEGImages/000038.jpg"
    # predictor.predict([sample_image_path]*10)
    # exit()

    print("Predict no label...")
    predictor.predict(test_path_no_label)

    predictor.evaluate(test_path)

    # eval_end = time.time()
    # print("The evaluation takes %.2f seconds." % (eval_end - train_end))

    # Load and reset num_gpus
    # new_predictor = MultiModalPredictor.load("./quick_start_tutorial_temp_save")
    # new_predictor.set_num_gpus(1)

    # Evaluate new predictor
    # print("Predict with loaded predictor...")
    # new_predictor.predict(test_path)
    # print("Predict no label with loaded predictor...")
    # new_predictor.predict(test_path_no_label)

    from autogluon.multimodal import download

    image_url = "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
    test_image = download(image_url)

    # create a input file for demo
    data = {"images": [{"id": 0, "width": -1, "height": -1, "file_name": test_image}], "categories": []}
    os.mkdir("input_data_for_demo")
    input_file = "input_data_for_demo/demo_annotation.json"
    with open(input_file, "w+") as f:
        json.dump(data, f)

    pred_test_image = predictor.predict(input_file)
    print(pred_test_image)


if __name__ == "__main__":
    tutorial_script_for_quick_start()
