from autogluon.multimodal import MultiModalPredictor

def test_coco_evaluation(checkpoint_name="faster_rcnn_r50_fpn_2x_coco"):
    predictor = MultiModalPredictor(
        label="rois",
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
        },
        pipeline="object_detection",
    )

    import time
    start = time.time()
    predictor.evaluate('coco17/annotations/instances_val2017.json')
    print("time usage: %.2f" % (time.time() - start))

if __name__ == "__main__":
    test_coco_evaluation("yolov3_mobilenetv2_320_300e_coco")

