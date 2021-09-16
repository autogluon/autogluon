from typing import Tuple, Union
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import copy

from .hp_ranges_cs import HyperparameterRanges_CS

RESOURCE_ATTR_PREFIX = 'RESOURCE_ATTR_'


class ExtendedConfiguration(object):
    """
    This class facilitates handling extended configs, which consist of a normal
    config and a resource attribute.

    The config space hp_ranges is extended by an additional resource
    attribute. Note that this is not a hyperparameter we optimize over,
    but it is under the control of the scheduler.
    Its allowed range is [1, resource_attr_range[1]], which can be larger than
    [resource_attr_range[0], resource_attr_range[1]]. This is because extended
    configs with resource values outside of resource_attr_range may arise (for
    example, in the early stopping context, we may receive data from
    epoch < resource_attr_range[0]).

    """
    def __init__(
            self, hp_ranges: HyperparameterRanges_CS, resource_attr_key: str,
            resource_attr_range: Tuple[int, int]):
        assert resource_attr_range[0] >= 1
        assert resource_attr_range[1] >= resource_attr_range[0]
        self.hp_ranges = hp_ranges
        self.resource_attr_key = resource_attr_key
        self.resource_attr_range = resource_attr_range
        # Extended configuration space including resource attribute
        config_space_ext = copy.deepcopy(hp_ranges.config_space)
        self.resource_attr_name = RESOURCE_ATTR_PREFIX + resource_attr_key
        # Allowed range: [1, resource_attr_range[1]]
        config_space_ext.add_hyperparameter(CSH.UniformIntegerHyperparameter(
            name=self.resource_attr_name, lower=1,
            upper=resource_attr_range[1]))
        self.hp_ranges_ext = HyperparameterRanges_CS(
            config_space_ext, name_last_pos=self.resource_attr_name)

    def get(self, config: CS.Configuration, resource: int) -> CS.Configuration:
        """
        Create extended config with resource added.

        :param config:
        :param resource:
        :return: Extended config
        """
        values = copy.deepcopy(config.get_dictionary())
        values[self.resource_attr_name] = resource
        return CS.Configuration(self.hp_ranges_ext.config_space, values=values)

    def remap_resource(
            self, config_ext: CS.Configuration, resource: int,
            as_dict: bool=False) -> Union[CS.Configuration, dict]:
        """
        Re-assigns resource value for extended config.

        :param config_ext: Extended config
        :param resource: New resource value
        :param as_dict: Return as dict?
        :return:
        """
        x_dct = copy.copy(config_ext.get_dictionary())
        x_dct[self.resource_attr_name] = resource
        if as_dict:
            return x_dct
        else:
            return CS.Configuration(
                self.hp_ranges_ext.config_space, values=x_dct)

    def remove_resource(
            self, config_ext: CS.Configuration,
            as_dict: bool=False) -> Union[CS.Configuration, dict]:
        """
        Strips away resource attribute and returns normal config

        :param config_ext: Extended config
        :param as_dict: Return as dict?
        :return: config_ext without resource attribute
        """
        x_dct = copy.copy(config_ext.get_dictionary())
        del x_dct[self.resource_attr_name]
        if as_dict:
            return x_dct
        else:
            return CS.Configuration(self.hp_ranges.config_space, values=x_dct)

    def from_dict(self, config_dct: dict) -> CS.Configuration:
        """
        Converts dict into CS.Configuration config (extended or normal, depending
        on whether the dict contains a resource attribute).

        :param config_dct:
        :return:
        """
        # Note: Here, the key for resource is resource_attr_key, not
        # resource_attr_name
        hp_ranges = self.hp_ranges_ext if self.resource_attr_key in config_dct \
            else self.hp_ranges
        return CS.Configuration(hp_ranges.config_space, values=config_dct)

    def split(self, config_ext: CS.Configuration, as_dict: bool=False) -> \
            (Union[CS.Configuration, dict], int):
        """
        Split extended config into normal config and resource value.

        :param config_ext: Extended config
        :param as_dict: Return config as dict?
        :return: (config, resource_value)
        """
        x_res = copy.copy(config_ext.get_dictionary())
        resource_value = int(x_res[self.resource_attr_name])
        del x_res[self.resource_attr_name]
        if not as_dict:
            x_res = CS.Configuration(self.hp_ranges.config_space, values=x_res)
        return x_res, resource_value

    def get_resource(self, config_ext: CS.Configuration) -> int:
        """
        :param config_ext: Extended config
        :return: Value of resource attribute
        """
        return int(config_ext.get_dictionary()[self.resource_attr_name])
