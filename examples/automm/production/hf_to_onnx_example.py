from autogluon.multimodal import MultiModalPredictor

import argparse
import numpy as np
import onnx
import onnxruntime as ort
import torch
from datasets import load_dataset
from torch import tensor


VALID_HF_INPUT = [
    "hf_text_text_token_ids",
    "hf_text_text_valid_length",
    "hf_text_text_segment_ids",
]

VALID_ONNX_INPUT = [
    "hf_text_text_token_ids",
    "hf_text_text_valid_length",
    "hf_text_text_segment_ids",  # comment this line for mpnet
]


def get_hf_model(predictor, device="cpu"):
    model = predictor._model
    model.to(device)
    model.eval()
    return model


def export_to_onnx(model, batch=None, onnx_path="dummy.onnx", verbose=False):
    torch.onnx.export(
        model,
        batch,
        onnx_path,
        opset_version=13,
        verbose=verbose,
        input_names=VALID_ONNX_INPUT,
        dynamic_axes={
            "hf_text_text_token_ids": {
                0: "batch_size",
                1: "sentence_length",
            },
            "hf_text_text_segment_ids": {
                0: "batch_size",
                1: "sentence_length",
            },
        },
    )
    print(f"ONNX model saved to {onnx_path}")


def load_onnx(onnx_path="dummy.onnx", print_model=False):
    # onnx load

    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    if print_model:
        print(onnx.helper.printable_graph(model_onnx.graph))

    return model_onnx


def get_onnx_result(onnx_batch, ort_sess):
    return ort_sess.run(None, onnx_batch)[0]


def get_hf_result(hf_batch, model):
    return model(hf_batch)["hf_text"]["features"].cpu().detach().numpy()


def compare_result(hf_batch, onnx_batch, model, ort_sess, print_embedding=False):
    hf_result = get_hf_result(hf_batch, model)
    onnx_result = get_onnx_result(onnx_batch, ort_sess)
    if print_embedding:
        print("hf_result:")
        print(hf_result)
        print("onnx_result:")
        print(onnx_result)
    print("distance between hf and onnx result:")
    print(np.linalg.norm(hf_result - onnx_result))


def get_hf_batch(
    raw_batch,
    device="cpu",
    valid_input=None,
):
    if valid_input is None:
        valid_input = VALID_HF_INPUT

    hf_batch = {}
    for k in raw_batch:
        if k in valid_input:
            hf_batch[k] = raw_batch[k].to(device).long()
    return hf_batch


def get_onnx_batch(
    raw_batch,
    valid_input=None,
):
    if valid_input is None:
        valid_input = VALID_ONNX_INPUT

    onnx_batch = {}
    for k in raw_batch:
        if k in valid_input:
            onnx_batch[k] = raw_batch[k].cpu().detach().numpy().astype(int)
    return onnx_batch


def main(
    data,
    train_batch,
    test_batches,
    args,
    valid_hf_input=None,
    valid_onnx_input=None,
):
    predictor = MultiModalPredictor(
        pipeline="feature_extraction",
        hyperparameters={
            "model.hf_text.checkpoint_name": args.checkpoint_name,
        },
    )
    model = get_hf_model(predictor, device=args.device)

    hf_train_batch = get_hf_batch(
        train_batch,
        device=args.device,
        valid_input=valid_hf_input,
    )
    onnx_train_batch = get_onnx_batch(
        train_batch,
        valid_input=valid_onnx_input,
    )

    predictor.export_onnx(data=data, batch=hf_train_batch, verbose=args.verbose, onnx_path=args.onnx_path)
    # export_to_onnx(model, hf_train_batch, verbose=verbose, onnx_path=onnx_path)

    load_onnx(args.onnx_path, True)

    ort_sess = ort.InferenceSession(args.onnx_path, providers=["CUDAExecutionProvider"])

    print("TRAIN batch result comparison:")
    compare_result(hf_train_batch, onnx_train_batch, model, ort_sess, print_embedding=args.print_embedding)

    print("TEST batch result comparison:")
    for test_batch in test_batches:
        hf_test_batch = get_hf_batch(
            test_batch,
            device=args.device,
            valid_input=valid_hf_input,
        )
        onnx_test_batch = get_onnx_batch(
            test_batch,
            valid_input=valid_onnx_input,
        )
        compare_result(hf_test_batch, onnx_test_batch, model, ort_sess, print_embedding=args.print_embedding)


if __name__ == "__main__":
    train_batch = {
        "hf_text_text_token_ids_column_sentence1": tensor(
            [[1, 7], [1, 8], [1, 8], [1, 8], [1, 6], [1, 9], [1, 8], [1, 8]], device="cuda:1"
        ),
        "hf_text_text_token_ids": tensor(
            [
                [101, 1037, 2158, 2003, 2652, 2858, 1012, 102, 0, 0, 0, 0, 0, 0],
                [101, 1037, 2158, 2003, 2652, 1037, 2858, 1012, 102, 0, 0, 0, 0, 0],
                [101, 1037, 2158, 2003, 2652, 1037, 2858, 1012, 102, 0, 0, 0, 0, 0],
                [101, 1037, 2158, 2003, 6276, 2019, 20949, 1012, 102, 0, 0, 0, 0, 0],
                [101, 1037, 2158, 2003, 9670, 1012, 102, 0, 0, 0, 0, 0, 0, 0],
                [101, 1037, 2158, 2003, 26514, 2330, 1037, 3869, 1012, 102, 0, 0, 0, 0],
                [101, 1037, 2158, 2003, 26514, 1037, 20856, 1012, 102, 0, 0, 0, 0, 0],
                [101, 1037, 2158, 2003, 2652, 1037, 2858, 1012, 102, 0, 0, 0, 0, 0],
            ],
            device="cuda:1",
            dtype=torch.int32,
        ),
        "hf_text_text_valid_length": tensor([8, 9, 9, 9, 7, 10, 9, 9], device="cuda:1"),
        "hf_text_text_segment_ids": tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            device="cuda:1",
            dtype=torch.int32,
        ),
    }

    test_batches = [
        {
            "hf_text_text_token_ids_column_sentence1": tensor(
                [[1, 9], [1, 10], [1, 7], [1, 8], [1, 8], [1, 8], [1, 10], [1, 8]], device="cuda:2"
            ),
            "hf_text_text_token_ids": tensor(
                [
                    [101, 1037, 3336, 25462, 3632, 2091, 1037, 7358, 1012, 102, 0, 0, 0, 0],
                    [101, 1037, 2158, 2003, 4823, 1998, 2652, 1037, 2858, 1012, 102, 0, 0, 0],
                    [101, 1037, 2158, 4491, 1037, 2450, 1012, 102, 0, 0, 0, 0, 0, 0],
                    [101, 1037, 2158, 2003, 4439, 1037, 2482, 1012, 102, 0, 0, 0, 0, 0],
                    [101, 1037, 2450, 2003, 6276, 2000, 11263, 1012, 102, 0, 0, 0, 0, 0],
                    [101, 1996, 2450, 2003, 20724, 2014, 2606, 1012, 102, 0, 0, 0, 0, 0],
                    [101, 2048, 29145, 2015, 2377, 1999, 2019, 2330, 2492, 1012, 102, 0, 0, 0],
                    [101, 1037, 2158, 2003, 6276, 1037, 14557, 1012, 102, 0, 0, 0, 0, 0],
                ],
                device="cuda:2",
                dtype=torch.int32,
            ),
            "hf_text_text_valid_length": tensor([10, 11, 8, 9, 9, 9, 11, 9], device="cuda:2"),
            "hf_text_text_segment_ids": tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                device="cuda:2",
                dtype=torch.int32,
            ),
        },
        {
            "hf_text_text_token_ids_column_sentence1": tensor(
                [[1, 8], [1, 6], [1, 8], [1, 8], [1, 6], [1, 11], [1, 8], [1, 13]], device="cuda:3"
            ),
            "hf_text_text_token_ids": tensor(
                [
                    [101, 1037, 2158, 2003, 26514, 2019, 20949, 1012, 102, 0, 0, 0, 0, 0],
                    [101, 1037, 2158, 2003, 5613, 1012, 102, 0, 0, 0, 0, 0, 0, 0],
                    [101, 1037, 2158, 2003, 5559, 1037, 9055, 1012, 102, 0, 0, 0, 0, 0],
                    [101, 1037, 2450, 2003, 26514, 20548, 2015, 1012, 102, 0, 0, 0, 0, 0],
                    [101, 1037, 2158, 2003, 4092, 1012, 102, 0, 0, 0, 0, 0, 0, 0],
                    [101, 1037, 2210, 2879, 2003, 4823, 1998, 2652, 1037, 2858, 1012, 102, 0, 0],
                    [101, 1037, 13170, 2003, 5742, 1999, 2300, 1012, 102, 0, 0, 0, 0, 0],
                    [101, 1037, 2402, 2450, 2003, 5128, 6293, 2545, 2035, 2058, 2014, 2227, 1012, 102],
                ],
                device="cuda:3",
                dtype=torch.int32,
            ),
            "hf_text_text_valid_length": tensor([9, 7, 9, 9, 7, 12, 9, 14], device="cuda:3"),
            "hf_text_text_segment_ids": tensor(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                device="cuda:3",
                dtype=torch.int32,
            ),
        },
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name", default="sentence-transformers/msmarco-MiniLM-L-12-v3", type=str)
    parser.add_argument("--onnx_path", default=None, type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print_embedding", action="store_true")
    args = parser.parse_args()

    if not args.onnx_path:
        args.onnx_path = args.checkpoint_name.replace("/", "_") + ".onnx"

    valid_hf_input = VALID_HF_INPUT
    valid_onnx_input = VALID_ONNX_INPUT

    data = load_dataset("wietsedv/stsbenchmark", split="test").to_pandas()

    main(
        data,
        train_batch,
        test_batches,
        args,
        valid_hf_input=valid_hf_input,
        valid_onnx_input=valid_onnx_input,
    )
