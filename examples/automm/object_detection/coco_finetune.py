from autogluon.multimodal import MultiModalPredictor

voc_train_path = "/media/data/datasets/voc/VOCdevkit/VOCCOCO/voc07_trainval.json"
voc_test_path = "/media/data/datasets/voc/VOCdevkit/VOCCOCO/voc07_test.json"
coco_train_path = "coco17/annotations/instances_val2017.json"
coco_test_path = "coco17/annotations/instances_val2017.json"


def voccoco_scratch(checkpoint_name="faster_rcnn_r50_fpn_2x_coco", train_path=voc_train_path, test_path=voc_test_path):
    num_classes = 20

    predictor = MultiModalPredictor(
        label="rois_label",
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        pipeline="object_detection",
        output_shape=num_classes,
    )

    predictor.evaluate(test_path)

    import time

    start = time.time()
    predictor.fit_coco(
        train_path,
        hyperparameters={
            "optimization.max_epochs": 10,
        },
    )
    predictor.evaluate(test_path)
    print("time usage: %.2f" % (time.time() - start))


def voccoco_finetune(
    checkpoint_name="faster_rcnn_r50_fpn_2x_coco", train_path=voc_train_path, test_path=voc_test_path
):
    num_classes = 20

    predictor = MultiModalPredictor(
        label="rois_label",
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        pipeline="object_detection",
        output_shape=num_classes,
    )

    import time

    start = time.time()
    predictor.fit_coco(
        train_path,
        hyperparameters={
            "optimization.learning_rate": 1e-3,
            "optimization.weight_decay": 1e-4,
            "optimization.max_epochs": 50,
            "env.per_gpu_batch_size": 2,
        },
    )
    fit_end = time.time()
    print("time usage for fit: %.2f" % (fit_end - start))

    predictor.evaluate(train_path)
    predictor.evaluate(test_path)
    print("time usage for eval: %.2f" % (time.time() - fit_end))


if __name__ == "__main__":
    # voccoco_finetune("yolov3_mobilenetv2_320_300e_coco", coco_test_path, coco_test_path)
    voccoco_finetune("yolov3_mobilenetv2_320_300e_coco")
    # voccoco_finetune("faster_rcnn_r50_fpn_2x_coco")
