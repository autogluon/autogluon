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

    # Init predictor
    predictor = MultiModalPredictor(
        problem_type="object_detection",
        sample_data_path=train_path,
        path=f"./AutogluonModels/{dataset_name}_bench_{presets}_tune_{uuid.uuid4()}",
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
