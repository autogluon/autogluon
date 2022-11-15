import argparse

from autogluon.multimodal import MultiModalPredictor

WATERCOLOR = {
    "train_path": "/media/code/detdata/DetBenchmark/watercolor/Annotations/train_cocoformat.json",
    "val_path": "/media/code/detdata/DetBenchmark/watercolor/Annotations/val_cocoformat.json",
    "test_path": "/media/code/detdata/DetBenchmark/watercolor/Annotations/test_cocoformat.json",
}

medium_quality_faster_train_hyperparameters = {
    "model.names": ["mmdet_image"],
    "model.mmdet_image.checkpoint_name": "yolov3_mobilenetv2_320_300e_coco",
    "env.eval_batch_size_ratio": 1,
    "env.precision": 32,
    "env.strategy": "ddp",
    "env.auto_select_gpus": False,  # Have to turn off for detection!
    "optimization.learning_rate": 1e-4,
    "optimization.lr_decay": 0.95,
    "optimization.lr_mult": 100,
    "optimization.lr_choice": "two_stages",
    "optimization.top_k": 1,
    "optimization.top_k_average_method": "best",
    "optimization.warmup_steps": 0.0,
    "optimization.patience": 10,
    "optimization.max_epochs": 10,
    "optimization.val_metric": "direct_loss",
    "env.num_gpus": 4,  # TODO: remove this in presets
    "env.per_gpu_batch_size": 32,  # TODO: remove this in presets
}

high_quality_fast_train_hyperparameters = {
    "model.names": ["mmdet_image"],
    "model.mmdet_image.checkpoint_name": "yolov3_d53_mstrain-416_273e_coco",
    "env.eval_batch_size_ratio": 1,
    "env.precision": 32,
    "env.strategy": "ddp",
    "env.auto_select_gpus": False,  # Have to turn off for detection!
    "optimization.learning_rate": 1e-5,
    "optimization.lr_decay": 0.95,
    "optimization.lr_mult": 100,
    "optimization.lr_choice": "two_stages",
    "optimization.top_k": 1,
    "optimization.top_k_average_method": "best",
    "optimization.warmup_steps": 0.0,
    "optimization.patience": 10,
    "optimization.max_epochs": 20,
    "optimization.val_metric": "map",
    "env.num_gpus": 4,  # TODO: remove this in presets
    "env.per_gpu_batch_size": 32,  # TODO: remove this in presets
}

higher_quality_hyperparameters = {
    "model.names": ["mmdet_image"],
    "model.mmdet_image.checkpoint_name": "vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco",
    "env.eval_batch_size_ratio": 1,
    "env.precision": 32,
    "env.strategy": "ddp",
    "env.auto_select_gpus": False,  # Have to turn off for detection!
    "optimization.learning_rate": 5e-6,
    "optimization.lr_decay": 0.95,
    "optimization.lr_mult": 100,
    "optimization.lr_choice": "two_stages",
    "optimization.top_k": 1,
    "optimization.top_k_average_method": "best",
    "optimization.warmup_steps": 0.0,
    "optimization.patience": 10,
    "optimization.max_epochs": 30,
    "optimization.val_metric": "map",
    "env.num_gpus": 4,  # TODO: remove this in presets
    "env.per_gpu_batch_size": 4,  # TODO: remove this in presets
}

best_quality_hyperparameters = {
    "model.names": ["mmdet_image"],
    "model.mmdet_image.checkpoint_name": "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco",
    "env.eval_batch_size_ratio": 1,
    "env.precision": 32,
    "env.strategy": "ddp",
    "env.auto_select_gpus": False,  # Have to turn off for detection!
    "optimization.learning_rate": 1e-5,
    "optimization.lr_decay": 0.95,
    "optimization.lr_mult": 100,
    "optimization.lr_choice": "two_stages",
    "optimization.top_k": 1,
    "optimization.top_k_average_method": "best",
    "optimization.warmup_steps": 0.0,
    "optimization.patience": 10,
    "optimization.max_epochs": 30,
    "optimization.val_metric": "map",
    "env.num_gpus": 4,  # TODO: remove this in presets
    "env.per_gpu_batch_size": 2,  # TODO: remove this in presets
}

ultra_quality_hyperparameters = {
    "model.names": ["mmdet_image"],
    "model.mmdet_image.checkpoint_name": "vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco",
    "env.eval_batch_size_ratio": 1,
    "env.precision": 32,
    "env.strategy": "ddp",
    "env.auto_select_gpus": False,  # Have to turn off for detection!
    "optimization.learning_rate": 1e-6,
    "optimization.lr_decay": 0.95,
    "optimization.lr_mult": 100,
    "optimization.lr_choice": "two_stages",
    "optimization.top_k": 1,
    "optimization.top_k_average_method": "best",
    "optimization.warmup_steps": 0.0,
    "optimization.patience": 40,
    "optimization.max_epochs": 50,
    "optimization.val_metric": "map",
    "env.num_gpus": 4,  # TODO: remove this in presets
    "env.per_gpu_batch_size": 2,  # TODO: remove this in presets
}

DATASET = {"watercolor": WATERCOLOR}
PRESETS = {
    "fast": medium_quality_faster_train_hyperparameters,
    "high": high_quality_fast_train_hyperparameters,
    "higher": higher_quality_hyperparameters,
    "best": best_quality_hyperparameters,
    "ultra": ultra_quality_hyperparameters,
}


def presets_finetune(
    dataset_paths,
    presets,
):

    train_path = dataset_paths["train_path"]
    val_path = dataset_paths["val_path"]
    test_path = dataset_paths["test_path"]

    predictor = MultiModalPredictor(
        hyperparameters=presets,
        problem_type="object_detection",
        sample_data_path=train_path,
    )

    import time

    start = time.time()
    predictor.fit(
        train_path,
        tuning_data=val_path,
    )
    fit_end = time.time()
    print("time usage for fit: %.2f" % (fit_end - start))

    predictor.evaluate(test_path)
    print("time usage for eval: %.2f" % (time.time() - fit_end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-p", "--presets", type=str)
    args = parser.parse_args()

    presets_finetune(
        dataset_paths=DATASET[args.dataset],
        presets=PRESETS[args.presets],
    )
