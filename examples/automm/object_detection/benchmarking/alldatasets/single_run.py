import argparse

from autogluon.multimodal import MultiModalPredictor

NUM_GPUS = -1
EPOCHS = 40
VAL_METRIC = "map"

def get_hp(short_name, lr_mode):
    if short_name == "centernet_r18":
        full_name = "centernet_resnet18_dcnv2_140e_coco"
        base_head_lr = 0.01
        batch_size = 32
    elif short_name == "yolov3_mv2":
        full_name = "yolov3_mobilenetv2_320_300e_coco"
        base_head_lr = 0.01
        batch_size = 32
    elif short_name == "yolov3_d53":
        full_name = "yolov3_d53_mstrain-416_273e_coco"
        base_head_lr = 0.001
        batch_size = 32
    elif short_name == "cascadercnn_r50":
        full_name = "cascade_rcnn_r50_fpn_20e_coco"
        base_head_lr = 0.0005
        batch_size = 4
    elif short_name == "vfnet_r50":
        full_name = "vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco"
        base_head_lr = 0.0005
        batch_size = 4
    elif short_name == "vfnet_x101":
        full_name = "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco"
        base_head_lr = 0.001
        batch_size = 2
    else:
        raise ValueError(f"Invalid checkpoint_name: {short_name}.")

    if "5" in str(base_head_lr):
        if lr_mode == "higher":
            lr = base_head_lr * 10
        elif lr_mode == "high":
            lr = base_head_lr * 2
        elif lr_mode == "med":
            lr = base_head_lr
        elif lr_mode == "low":
            lr = base_head_lr / 5
        elif lr_mode == "lower":
            lr = base_head_lr / 10
        else:
            raise ValueError(f"Invalid lr_mode: {lr_mode}")
    elif "1" in str(base_head_lr):
        if lr_mode == "higher":
            lr = base_head_lr * 10
        elif lr_mode == "high":
            lr = base_head_lr * 5
        elif lr_mode == "med":
            lr = base_head_lr
        elif lr_mode == "low":
            lr = base_head_lr / 2
        elif lr_mode == "lower":
            lr = base_head_lr / 10
        else:
            raise ValueError(f"Invalid lr_mode: {lr_mode}")
    else:
        raise ValueError(f"Invalid base_head_lr {base_head_lr}")

    print(f"HP settings: \n  model: {full_name}\n  lr: {lr}\n  batch_size: {batch_size}")

    return full_name, lr, batch_size

def get_data_path(dataset_name):
    if dataset_name == "KITTI":
        return (
            "/media/code/datasets/detection/KITTI/Annotations/train_cocoformat.json",
            "/media/code/datasets/detection/KITTI/Annotations/val_cocoformat.json",
            "/media/code/datasets/detection/KITTI/Annotations/test_cocoformat.json",
        )
    elif dataset_name == "Kitchen":
        return (
            "/media/code/detdata/DetBenchmark/Kitchen/Annotations/train_cocoformat.json",
            None,
            "/media/code/detdata/DetBenchmark/Kitchen/Annotations/test_cocoformat.json",
        )
    elif dataset_name == "LISA":
        return (
            "/media/code/detdata/DetBenchmark/LISA/Annotations/train_cocoformat.json",
            None,
            "/media/code/detdata/DetBenchmark/LISA/Annotations/test_cocoformat.json",
        )
    elif dataset_name == "clipart":
        return (
            "/media/code/detdata/DetBenchmark/clipart/Annotations/train_cocoformat.json",
            None,
            "/media/code/detdata/DetBenchmark/clipart/Annotations/test_cocoformat.json",
        )
    elif dataset_name == "comic":
        return (
            "/media/code/detdata/DetBenchmark/comic/Annotations/train_cocoformat.json",
            None,
            "/media/code/detdata/DetBenchmark/comic/Annotations/test_cocoformat.json",
        )
    elif dataset_name == "deeplesion":
        return (
            "/media/code/detdata/DetBenchmark/deeplesion/Annotations/train_cocoformat.json",
            "/media/code/detdata/DetBenchmark/deeplesion/Annotations/val_cocoformat.json",
            "/media/code/detdata/DetBenchmark/deeplesion/Annotations/test_cocoformat.json",
        )
    elif dataset_name == "dota":
        return (
            "/media/code/detdata/DetBenchmark/dota/Annotations/train_cocoformat.json",
            None,
            "/media/code/detdata/DetBenchmark/dota/Annotations/val_cocoformat.json",
        )
    elif dataset_name == "watercolor":
        return (
            "/media/code/detdata/DetBenchmark/watercolor/Annotations/train_cocoformat.json",
            None,
            "/media/code/detdata/DetBenchmark/watercolor/Annotations/test_cocoformat.json",
        )
    elif dataset_name == "widerface":
        return (
            "/media/code/detdata/DetBenchmark/widerface/Annotations/train_cocoformat.json",
            None,
            "/media/code/detdata/DetBenchmark/widerface/Annotations/val_cocoformat.json",
        )
    elif dataset_name == "VOC":
        return (
            "/media/code/autogluon/examples/automm/object_detection/VOCdevkit/VOCCOCO/voc0712_train.json",
            "/media/code/autogluon/examples/automm/object_detection/VOCdevkit/VOCCOCO/voc07_val.json",
            "/media/code/autogluon/examples/automm/object_detection/VOCdevkit/VOCCOCO/voc07_test.json",
        )
    elif dataset_name == "pothole":
        return (
            "/media/code/autogluon/examples/automm/object_detection/pothole/pothole/Annotations/usersplit_train_cocoformat.json",
            "/media/code/autogluon/examples/automm/object_detection/pothole/pothole/Annotations/usersplit_val_cocoformat.json",
            "/media/code/autogluon/examples/automm/object_detection/pothole/pothole/Annotations/usersplit_test_cocoformat.json",
        )
    else:
        raise ValueError(f"Invalid dataset_name: {dataset_name}")


def single_run(
    dataset_name,
    checkpoint_name,
    lr,
    per_gpu_batch_size,
):
    train_path, val_path, test_path = get_data_path(dataset_name)

    predictor = MultiModalPredictor(
        label="label",
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": NUM_GPUS,
            "optimization.val_metric": VAL_METRIC,
        },
        problem_type="object_detection",
        sample_data_path=train_path,
    )

    import time

    start = time.time()
    predictor.fit(
        train_path,
        tuning_data=val_path,
        hyperparameters={
            "optimization.learning_rate": lr / 100,  # we use two stage and lr_mult=100 for detection
            "optimization.max_epochs": EPOCHS,
            "optimization.patience": 10,
            "env.per_gpu_batch_size": per_gpu_batch_size,  # decrease it when model is large
        },
    )
    fit_end = time.time()
    print("time usage for fit: %.2f" % (fit_end - start))

    predictor.evaluate(test_path)
    print("time usage for eval: %.2f" % (time.time() - fit_end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str)
    parser.add_argument("-c", "--short_checkpoint_name", type=str)
    parser.add_argument("-l", "--lr_mode", type=str)
    args = parser.parse_args()

    full_checkpoint_name, lr, batch_size = get_hp(args.short_checkpoint_name, args.lr_mode)

    single_run(
        dataset_name=args.dataset_name,
        checkpoint_name=full_checkpoint_name,
        lr=lr,
        per_gpu_batch_size=batch_size,
    )
