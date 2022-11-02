import argparse
import os.path

from autogluon.multimodal import MultiModalPredictor


def load_and_evaluate(
    load_path,
    test_path,
):
    predictor = MultiModalPredictor.load(load_path)

    if os.path.isdir(load_path):
        predictor.set_num_gpus(num_gpus=1)

    import time

    start = time.time()
    result = predictor.evaluate(test_path)
    print("time usage: %.2f" % (time.time() - start))
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--test_path", type=str)
    args = parser.parse_args()

    load_and_evaluate(
        load_path=args.load_path,
        test_path=args.test_path,
    )
