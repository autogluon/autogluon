from autogluon.multimodal import MultiModalPredictor


def detection_evaluation(
    checkpoint_name="faster_rcnn_r50_fpn_2x_coco", anno_path="coco17/annotations/instances_val2017.json"
):
    predictor = MultiModalPredictor(
        label="rois_label",
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        pipeline="object_detection",
    )

    import time

    start = time.time()
    predictor.evaluate(anno_path)
    print("time usage: %.2f" % (time.time() - start))


if __name__ == "__main__":
    detection_evaluation("yolov3_mobilenetv2_320_300e_coco")
    # detection_evaluation("faster_rcnn_r50_fpn_1x_voc0712", "VOCdevkit/VOCCOCO/voc07_test.json")
