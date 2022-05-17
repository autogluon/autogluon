import pytest
from autogluon.text.automm.utils import apply_omegaconf_overrides, parse_dotlist_conf
from omegaconf import OmegaConf


@pytest.mark.parametrize('data,expected',
                         [
                             ('aaa=a bbb=b ccc=c', {'aaa': 'a', 'bbb': 'b', 'ccc': 'c'}),
                             ('a.a.aa=b b.b.bb=c', {'a.a.aa': 'b', 'b.b.bb': 'c'}),
                             ('a.a.aa=1 b.b.bb=100', {'a.a.aa': '1', 'b.b.bb': '100'}),
                             (['a.a.aa=1', 'b.b.bb=100'], {'a.a.aa': '1', 'b.b.bb': '100'})
                         ])
def test_parse_dotlist_conf(data, expected):
    assert parse_dotlist_conf(data) == expected


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

    with pytest.raises(KeyError):
        new_conf3 = apply_omegaconf_overrides(conf, {'a.aa.aaaaaa': [1, 3, 5, 7], 'a.aa.bbb': 4})
