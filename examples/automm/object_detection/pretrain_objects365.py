import os
import time

from autogluon.multimodal import MultiModalPredictor


def main():
    data_dir = "/media/code/datasets/object365"
    train_path = os.path.join(data_dir, "train", "annotations", "zhiyuan_objv2_train.json")
    val_path = os.path.join(data_dir, "val", "annotations", "zhiyuan_objv2_val.json")

    checkpoint_name = "yolox_l_8x8_300e_coco"
    num_gpus = 3

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
            "optim.val_metric": "map",
        },
        problem_type="object_detection",
        sample_data_path=train_path,
    )

    start = time.time()
    predictor.fit(
        train_path,
        tuning_data=val_path,
        max_num_tuning_data=5000,
        hyperparameters={
            "optim.lr": 1e-3,  # we use two stage and detection head has 100x lr
            "optim.lr_decay": 0.9,
            "optim.lr_mult": 1,
            "optim.max_epochs": 12,
            # "optim.max_steps": 180000,
            "optim.warmup_steps": 0.1,
            "optim.patience": 1000,
            "optim.val_check_interval": 0.25,
            "optim.check_val_every_n_epoch": 1,
            "optim.top_k": 20,
            "env.per_gpu_batch_size": 6,  # decrease it when model is large
        },
        clean_ckpts=False,
    )
    end = time.time()

    print("This finetuning takes %.2f seconds." % (end - start))


if __name__ == "__main__":
    main()
