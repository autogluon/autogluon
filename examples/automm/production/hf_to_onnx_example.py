from autogluon.multimodal import MultiModalPredictor

import numpy as np
import onnx
import onnxruntime as ort
import torch


VALID_INPUT = [
    "hf_text_text_token_ids_column_sentence1",
    "hf_text_text_token_ids",
    "hf_text_text_valid_length",
    "hf_text_text_segment_ids",
]


def get_hf_model(predictor, device="cpu"):
    model = predictor._model
    model.to(device)
    model.eval()
    return model


def export_to_onnx(model, batch, onnx_path="dummy.onnx", verbose=False):
    torch.onnx.export(
        model,
        batch,
        onnx_path,
        opset_version=13,
        verbose=verbose,
        input_names=[
            "hf_text_text_token_ids_column_sentence1",
            "hf_text_text_token_ids",
            "hf_text_text_valid_length",
            "hf_text_text_segment_ids",
        ],
        dynamic_axes={
            "hf_text_text_token_ids_column_sentence1": {
                0: "batch_size",
                1: "sentence_length",
            },
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
    return model(hf_batch)["hf_text"]["column_features"]["features"]["sentence1"].cpu().detach().numpy()


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
        valid_input = VALID_INPUT

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
        valid_input = VALID_INPUT

    onnx_batch = {}
    for k in raw_batch:
        if k in valid_input:
            onnx_batch[k] = raw_batch[k].cpu().detach().numpy().astype(int)
    return onnx_batch


def main(
    train_batch,
    test_batches,
    device="cpu",
    verbose=False,
    valid_input=None,
    onnx_path="dummy.onnx",
    print_embedding=False,
):
    predictor = MultiModalPredictor(pipeline="feature_extraction")
    model = get_hf_model(predictor, device=device)

    hf_train_batch = get_hf_batch(
        train_batch,
        device=device,
        valid_input=valid_input,
    )
    onnx_train_batch = get_onnx_batch(
        train_batch,
        valid_input=valid_input,
    )

    export_to_onnx(model, hf_train_batch, verbose=verbose, onnx_path=onnx_path)

    ort_sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])

    print("TRAIN batch result comparison:")
    compare_result(hf_train_batch, onnx_train_batch, model, ort_sess, print_embedding=print_embedding)

    print("TEST batch result comparison:")
    for test_batch in test_batches:
        hf_test_batch = get_hf_batch(
            test_batch,
            device=device,
            valid_input=valid_input,
        )
        onnx_test_batch = get_onnx_batch(
            test_batch,
            valid_input=valid_input,
        )
        compare_result(hf_test_batch, onnx_test_batch, model, ort_sess, print_embedding=print_embedding)


if __name__ == "__main__":
    train_batch = {
        "hf_text_text_token_ids_column_sentence1": torch.tensor(
            [
                [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            ],
        ),
        "hf_text_text_token_ids": torch.tensor(
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
        ),
        "hf_text_text_valid_length": torch.tensor([8, 9, 9, 9, 7, 10, 9, 9]),
        "hf_text_text_segment_ids": torch.tensor(
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
        ),
    }

    test_batches = [
        {
            "hf_text_text_token_ids_column_sentence1": torch.tensor(
                [
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                ],
            ),
            "hf_text_text_token_ids": torch.tensor(
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
            ),
            "hf_text_text_valid_length": torch.tensor([10, 11, 8, 9, 9, 9, 11, 9]),
            "hf_text_text_segment_ids": torch.tensor(
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
            ),
        }
    ]

    device = "cpu"
    verbose = False
    valid_input = VALID_INPUT
    onnx_path = "dummy.onnx"
    print_embedding = False

    main(
        train_batch,
        test_batches,
        device=device,
        verbose=verbose,
        valid_input=valid_input,
        onnx_path=onnx_path,
        print_embedding=print_embedding,
    )
