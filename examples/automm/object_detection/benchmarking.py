import argparse
import os
import time
import uuid

from autogluon.multimodal import MultiModalPredictor


def main(benchmark_root, dataset_name, presets, seed):
    train_path = os.path.join(benchmark_root, dataset_name, "annotations", "train_train.json")
    val_path = os.path.join(benchmark_root, dataset_name, "annotations", "train_val.json")
    if not os.path.exists(val_path):
        val_path = None
        print(f"Validation path {val_path} does not exist.")
    test_path = os.path.join(benchmark_root, dataset_name, "annotations", "test.json")

    # Init predictor
    predictor = MultiModalPredictor(
        problem_type="object_detection",
        sample_data_path=train_path,
        path=f"./AutogluonModels/{dataset_name}_bench_{presets}_seed_{seed}_tune_{uuid.uuid4()}",
        presets=presets,
    )

    # Fit
    start = time.time()
    predictor.fit(
        train_path,
        tuning_data=val_path,
        seed=seed,
    )
    train_end = time.time()
    print("The finetuning takes %.2f seconds." % (train_end - start))

    predictor.evaluate(test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--benchmark_root", default="./data/AutoMLDetBench", type=str)
    parser.add_argument("-d", "--dataset_name", default=None, type=str)
    parser.add_argument("-p", "--presets", default="best_quality", type=str)
    parser.add_argument("-s", "--seed", default=0, type=int)
    args = parser.parse_args()

    main(
        benchmark_root=args.benchmark_root,
        dataset_name=args.dataset_name,
        presets=args.presets,
        seed=args.seed,
    )
