from autogluon.text.automm.utils import apply_omegaconf_overrides
from omegaconf import OmegaConf


def test_apply_omegaconf_overrides():
    conf = OmegaConf.from_dotlist(["a.aa.aaa=[1, 2, 3, 4]",
                                   "a.aa.bbb=2",
                                   "a.bb.aaa='100'",
                                   "a.bb.bbb=4"])
    overrides = 'a.aa.aaa=[1, 3, 5] a.aa.bbb=3'
    new_conf = apply_omegaconf_overrides(conf, overrides.split())
    assert new_conf.a.aa.aaa == [1, 3, 5]
    assert new_conf.a.aa.bbb == 3
    new_conf2 = apply_omegaconf_overrides(conf, {'a.aa.aaa': [1, 3, 5, 7], 'a.aa.bbb': 4})
    assert new_conf2.a.aa.aaa == [1, 3, 5, 7]
    assert new_conf2.a.aa.bbb == 4
