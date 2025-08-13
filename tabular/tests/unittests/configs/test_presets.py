import unittest

from autogluon.core.utils.decorators import apply_presets
from autogluon.tabular.configs.presets_configs import tabular_presets_alias, tabular_presets_dict


class TestPresets(unittest.TestCase):
    def test_presets(self):
        @apply_presets(tabular_presets_dict, tabular_presets_alias)
        def get_presets(presets=None, **kwargs):
            return kwargs

        # assert presets are applying their values correctly
        for preset_real in tabular_presets_dict:
            assert get_presets(presets=preset_real) == tabular_presets_dict[preset_real]

        # assert preset aliases are applying their values correctly
        for preset_alias, preset_real in tabular_presets_alias.items():
            assert get_presets(presets=preset_alias) == tabular_presets_dict[preset_real]

        # assert the quality presets exist
        for preset in ["extreme_quality", "best_quality", "high_quality", "good_quality", "medium_quality"]:
            assert preset in tabular_presets_dict
