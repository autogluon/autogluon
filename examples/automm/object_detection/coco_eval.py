from autogluon.multimodal import MultiModalPredictor

def test_coco_evaluation(checkpoint_name="faster_rcnn_r50_fpn_2x_coco"):
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
    predictor.evaluate('coco17/annotations/instances_val2017.json')
    print("time usage: %.2f" % (time.time() - start))

def test_voc_evaluation(checkpoint_name="faster_rcnn_r50_fpn_1x_voc0712"):
    predictor = MultiModalPredictor(
        label="rois_label",
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": 1,
            "env.per_gpu_batch_size": 1,
        },
        pipeline="object_detection",
    )

    import time
    start = time.time()
    predictor.evaluate('VOCdevkit/VOCCOCO/voc07_test.json')
    print("time usage: %.2f" % (time.time() - start))

def test_voc_load():
    from autogluon.multimodal.utils import from_coco, from_voc
    cocofmt_voc = from_coco('VOCdevkit/VOCCOCO/voc07_test.json')
    print(len(cocofmt_voc))
    print(cocofmt_voc[:2])
    print(cocofmt_voc["image"][0])
    print(cocofmt_voc["rois"][0])

    voc = from_voc("VOCdevkit/VOC2007")
    print(len(voc))
    print(voc[:2])
    print(cocofmt_voc["image"][0])
    print(voc["rois"][0])



if __name__ == "__main__":
    # test_coco_evaluation("yolov3_mobilenetv2_320_300e_coco")
    test_voc_evaluation("faster_rcnn_r50_fpn_1x_voc0712")
    # test_voc_load()
