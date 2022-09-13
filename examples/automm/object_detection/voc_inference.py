import time

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import from_voc


def test_voc_inference(checkpoint_name="faster_rcnn_r50_fpn_2x_coco"):
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        pipeline="object_detection",
    )

    df = from_voc("VOCdevkit/VOC2007")[:100][["image"]]

    start = time.time()
    pred = predictor.predict(df, as_pandas=False)  # TODO: disable as_pandas flag for detection
    print("time usage: %.2f" % (time.time() - start))

    assert len(pred) == len(df)
    assert len(pred[0]) == 80  # COCO has 80 classes
    assert pred[0][0].ndim == 2  # two dimensions, (# of proposals, 5)


if __name__ == "__main__":
    test_voc_inference()
