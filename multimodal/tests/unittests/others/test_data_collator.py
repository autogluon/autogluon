import torch

from autogluon.multimodal.data.collator import DictCollator, ListCollator, PadCollator, StackCollator, TupleCollator


def test_stack():
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9]
    ret = StackCollator()((a, b, c))
    assert torch.all(ret == torch.as_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


def test_pad():
    a = [1, 2, 3]
    b = [4, 5]
    c = [7]
    ret = PadCollator(pad_val=1)((a, b, c))
    assert torch.all(ret == torch.as_tensor([[1, 2, 3], [4, 5, 1], [7, 1, 1]]))

    ret = PadCollator(pad_val=1, round_to=2)((a, b, c))
    assert torch.all(ret == torch.as_tensor([[1, 2, 3, 1], [4, 5, 1, 1], [7, 1, 1, 1]]))

    ret = PadCollator(pad_val=1, max_length=4)((a, b, c))
    assert torch.all(ret == torch.as_tensor([[1, 2, 3, 1], [4, 5, 1, 1], [7, 1, 1, 1]]))

    ret, valid_lengths = PadCollator(pad_val=1, ret_length=True)((a, b, c))
    assert torch.all(valid_lengths == torch.as_tensor([3, 2, 1]))

    a = [[1, 4], [2, 5], [3, 6]]
    b = [[7], [8], [9]]
    c = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]

    ret = PadCollator(axis=1)((a, b, c))
    assert torch.all(
        ret
        == torch.as_tensor(
            [
                [[1, 4, 0], [2, 5, 0], [3, 6, 0]],
                [[7, 0, 0], [8, 0, 0], [9, 0, 0]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            ]
        )
    )


def test_tuple():
    a = ([1, 2, 3, 4], 0)
    b = ([5, 7], 1)

    ret_1, ret_2 = TupleCollator(PadCollator(), StackCollator())((a, b))

    assert torch.all(ret_1 == torch.as_tensor([[1, 2, 3, 4], [5, 7, 0, 0]]))
    assert torch.all(ret_2 == torch.as_tensor([0, 1]))


def test_list():
    a = ([1, 2, 3, 4], "id_0")
    b = ([5, 7, 2, 5], "id_1")
    c = ([1, 2, 3, 4], "id_2")
    _, l = TupleCollator(StackCollator(), ListCollator())((a, b, c))
    assert l == ["id_0", "id_1", "id_2"]


def test_dict():
    a = {"data": [1, 2, 3, 4, 5], "label": 0}
    b = {"data": [5, 7], "label": 1}
    c = {"data": [1, 2, 3], "label": 0}

    collate_fn = DictCollator({"data": PadCollator(), "label": StackCollator()})
    sample = collate_fn((a, b, c))

    assert torch.all(sample["data"] == torch.as_tensor([[1, 2, 3, 4, 5], [5, 7, 0, 0, 0], [1, 2, 3, 0, 0]]))
    assert torch.all(sample["label"] == torch.as_tensor([0, 1, 0]))
