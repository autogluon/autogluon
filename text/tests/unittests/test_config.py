import pytest
from autogluon.text.text_prediction.config import CfgNode

def test_to_flat_dict():
    cfg = CfgNode()
    cfg.a = '1'
    cfg.b = '2'
    cfg.c = CfgNode()
    cfg.c.d = '3'
    cfg.c.e = CfgNode()
    cfg.c.e.f = [(1, 2), (3, 4)]
    cfg.c.e.g = 1

    flat_dict = cfg.to_flat_dict()
    assert flat_dict == {'a': '1', 'b': '2', 'c.d': '3',
                         'c.e.f': [(1, 2), (3, 4)], 'c.e.g': 1}
