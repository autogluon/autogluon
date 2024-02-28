import argparse
import uuid
import os
import time

from autogluon.multimodal import MultiModalPredictor

BENCH_ROOT = "/media/ag/data/AutoMLDetBench"

def main(dataset_name, has_val, presets, lr_mult):
    train_path = os.path.join(BENCH_ROOT, dataset_name, "annotations", "train_train.json")
    val_path = os.path.join(BENCH_ROOT, dataset_name, "annotations", "train_val.json")
    if not os.path.exists(val_path):
        val_path = None
        print(f"Validation path {val_path} does not exist.")
    test_path = os.path.join(BENCH_ROOT, dataset_name, "annotations", "test.json")

    hyperparameters = {
        #"model.names": ["mmdet_image"],
        #"model.mmdet_image.checkpoint_name": "yolox_l",
        #"env.eval_batch_size_ratio": 1,
        #"env.precision": 32,
        #"env.strategy": "ddp",
        #"env.auto_select_gpus": True,
        #"env.num_gpus": -1,
        "env.batch_size": 32,
        "env.per_gpu_batch_size": 1,
        "env.num_workers": 2,
        #"optimization.optim_type": "sgd",
        "optimization.learning_rate": 1e-5,
        #"optimization.lr_decay": 0.95,
        "optimization.weight_decay": 1e-4,
        "optimization.lr_mult": 10,
        "optimization.lr_choice": "two_stages",  # TODO: changed
        "optimization.gradient_clip_val": 0.1,
        #"optimization.top_k": 1,
        #"optimization.top_k_average_method": "best",
        #"optimization.warmup_steps": 0.,
        #"optimization.patience": 10,
        #"optimization.val_check_interval": 1.0,
        #"optimization.check_val_evdery_n_epoch": 10,
        "optimization.max_epochs": 60,
    }

    # Init predictor
    predictor = MultiModalPredictor(
        #hyperparameters={"env.num_gpus": -1, "optimization.lr_mult": lr_mult},  # benchmarking on single GPU
        hyperparameters=hyperparameters,  # benchmarking on single GPU
        problem_type="object_detection",
        sample_data_path=train_path,
        #path=f"./{dataset_name}_bench_lrm{lr_mult}",
        path=f"./AutogluonModels/{dataset_name}_bench_{presets}_tune_{uuid.uuid4()}",
        #path=f"./AutogluonModels/{dataset_name}_tune_{uuid.uuid4()}",
        presets=presets,
    )

    # Fit
    start = time.time()
    predictor.fit(
        train_path,
        tuning_data=val_path,
    )
    train_end = time.time()
    print("The finetuning takes %.2f seconds." % (train_end - start))

    predictor.evaluate(test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", default=None, type=str)
    parser.add_argument("-p", "--presets", default="best_quality", type=str)
    parser.add_argument("-v", "--has_val", action="store_true")
    parser.add_argument("-m", "--lr_mult", default=100, type=int)
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        presets=args.presets,
        has_val=args.has_val,
        lr_mult=args.lr_mult,
    )
