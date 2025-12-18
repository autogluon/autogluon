import unittest

from autogluon.common.utils.decorators import apply_presets


class TestPresets(unittest.TestCase):
    def test_presets(self):
        presets_dict = dict(preset_1=dict(a=2, b=3))

        @apply_presets(presets_dict, None)
        def get_presets(**kwargs):
            return kwargs

        # assert no presets works
        out = get_presets()
        assert len(out) == 0

        # assert no presets works with user-specified values
        out = get_presets(a=5)
        assert out["a"] == 5
        assert len(out) == 1

        # assert ValueError raised if unknown preset
        self.assertRaises(ValueError, get_presets, presets="invalid_preset")

        # assert presets == None works
        out = get_presets(presets=None)
        assert out["presets"] is None
        assert len(out) == 1

        # assert presets as str works
        out = get_presets(presets="preset_1")
        assert out["presets"] == "preset_1"
        assert out["a"] == 2
        assert out["b"] == 3
        assert len(out) == 3

        # assert presets as list works
        out = get_presets(presets=["preset_1"])
        assert out["presets"] == ["preset_1"]
        assert out["a"] == 2
        assert out["b"] == 3
        assert len(out) == 3

        # assert custom preset works
        custom_preset = dict(a=4, c=7)
        out = get_presets(presets=custom_preset)
        assert out["presets"] == custom_preset
        assert out["a"] == 4
        assert out["c"] == 7
        assert len(out) == 3

        # assert that multiple presets can be specified, and later ones overwrite earlier ones in shared keys
        out = get_presets(presets=["preset_1", custom_preset])
        assert out["presets"] == ["preset_1", custom_preset]
        assert out["a"] == 4
        assert out["b"] == 3
        assert out["c"] == 7
        assert len(out) == 4

        # assert ValueError raised if unknown preset in list of presets
        self.assertRaises(ValueError, get_presets, presets=["preset_1", "invalid_preset"])

        # assert that multiple presets can be specified, and later ones overwrite earlier ones in shared keys, but user-specified keys override presets
        out = get_presets(a=1, presets=["preset_1", custom_preset], d=None)
        assert out["presets"] == ["preset_1", custom_preset]
        assert out["a"] == 1
        assert out["b"] == 3
        assert out["c"] == 7
        assert out["d"] is None
        assert len(out) == 5

        presets_alias_dict = dict(
            preset_1_alias="preset_1",
            preset_invalid_alias="invalid_preset",
        )

        @apply_presets(presets_dict, presets_alias_dict)
        def get_presets(**kwargs):
            return kwargs

        # assert preset alias works
        out = get_presets(presets="preset_1_alias")
        assert out["presets"] == "preset_1_alias"
        assert out["a"] == 2
        assert out["b"] == 3
        assert len(out) == 3

        # assert ValueError raised if alias points to invalid preset
        self.assertRaises(ValueError, get_presets, presets="preset_invalid_alias")
