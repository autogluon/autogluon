import time

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import from_coco, from_voc


# TODO: update inference API
def test_inference(dataset, checkpoint_name):
    assert dataset in ["coco", "voc"]

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        problem_type="object_detection",
    )

    if dataset == "coco":
        df = from_coco("coco17/annotations/instances_val2017.json")[:10][["image"]]
    elif dataset == "voc":
        df = from_voc("VOCdevkit/VOC2007")[:10][["image"]]

    start = time.time()
    pred = predictor.predict(df, as_pandas=False)  # TODO: disable as_pandas flag for detection
    print("time usage: %.2f" % (time.time() - start))

    assert len(pred) == len(df)

    assert len(pred[0]) == 80 if dataset == "coco" else 20  # COCO has 80 classes, VOC has 20 classes
    assert pred[0][0].ndim == 2  # two dimensions, (# of proposals, 5)


"""
VOC configs are not supported in mmcv.
And currently for voc inference,
we only support checkpoint_name="faster_rcnn_r50_fpn_1x_voc0712"
"""


def test_voc_inference(checkpoint_name="faster_rcnn_r50_fpn_1x_voc0712"):
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        problem_type="object_detection",
    )

    df = from_voc("VOCdevkit/VOC2007")[:10][["image"]]

    start = time.time()
    pred = predictor.predict(df, as_pandas=False)  # TODO: disable as_pandas flag for detection
    print("time usage: %.2f" % (time.time() - start))

    assert len(pred) == len(df)
    assert len(pred[0]) == 20  # VOC has 20 classes
    assert pred[0][0].ndim == 2  # two dimensions, (# of proposals, 5)


if __name__ == "__main__":
    # test coco inference
    test_inference("coco", "faster_rcnn_r50_fpn_2x_coco")

    # test voc inference
    # VOC configs are not supported in mmcv.
    # So currently for voc inference,
    # we only support checkpoint_name="faster_rcnn_r50_fpn_1x_voc0712"
    test_inference("voc", "faster_rcnn_r50_fpn_1x_voc0712")
