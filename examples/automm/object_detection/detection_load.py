import argparse

from autogluon.multimodal import MultiModalPredictor


def load_and_evaluate(
    load_path,
    test_path="/media/data/datasets/voc/VOCdevkit/VOCCOCO/voc07_test.json",
):
    predictor = MultiModalPredictor.load(load_path)

    import time

    start = time.time()
    result = predictor.evaluate(test_path, eval_tool="torchmetrics")
    print("time usage: %.2f" % (time.time() - start))
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--test_path", default="/media/data/datasets/voc/VOCdevkit/VOCCOCO/voc07_test.json", type=str)
    args = parser.parse_args()

    load_and_evaluate(
        load_path=args.load_path,
        test_path=args.test_path,
    )
