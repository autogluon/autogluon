import argparse

from autogluon.multimodal import MultiModalPredictor


def detection_evaluation(
    checkpoint_name="yolov3_mobilenetv2_320_300e_coco", test_path="coco17/annotations/instances_val2017.json",
        num_gpus=1,
):
    predictor = MultiModalPredictor(
        label="rois_label",
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        pipeline="object_detection",
    )

    import time

    start = time.time()
    result = predictor.evaluate(test_path)
    print("time usage: %.2f" % (time.time() - start))
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="coco17/annotations/instances_val2017.json", type=str)
    parser.add_argument("--checkpoint_name", default="yolov3_mobilenetv2_320_300e_coco", type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    args = parser.parse_args()

    detection_evaluation(
        test_path=args.test_path,
        checkpoint_name=args.checkpoint_name,
        num_gpus=args.num_gpus,
    )
    # detection_evaluation("faster_rcnn_r50_fpn_1x_voc0712", "VOCdevkit/VOCCOCO/voc07_test.json")
