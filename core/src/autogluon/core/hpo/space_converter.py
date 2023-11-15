from abc import ABC, abstractmethod

from ray import tune

from autogluon.common import space as ag_space


class RaySpaceConverter(ABC):
    @property
    @abstractmethod
    def space_type(self):
        """Type of the converter"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convert(space):
        """Convert the search space to ray search space"""
        raise NotImplementedError


class RayCategoricalSpaceConverter(RaySpaceConverter):
    @property
    def space_type(self):
        return ag_space.Categorical.__name__

    @staticmethod
    def convert(space):
        assert isinstance(space, ag_space.Categorical)
        return tune.choice(space.data)


class RayRealSpaceConverter(RaySpaceConverter):
    @property
    def space_type(self):
        return ag_space.Real.__name__

    @staticmethod
    def convert(space):
        assert isinstance(space, ag_space.Real)
        if space.log:
            ray_space = tune.loguniform(space.lower, space.upper)
        else:
            ray_space = tune.uniform(space.lower, space.upper)
        return ray_space


class RayIntSpaceConverter(RaySpaceConverter):
    @property
    def space_type(self):
        return ag_space.Int.__name__

    @staticmethod
    def convert(space):
        assert isinstance(space, ag_space.Int)
        return tune.randint(space.lower, space.upper + 1)


class RayBoolSpaceConverter(RayIntSpaceConverter):
    @property
    def space_type(self):
        return ag_space.Bool.__name__


class RaySpaceConverterFactory:
    __supported_converters = [
        RayCategoricalSpaceConverter,
        RayRealSpaceConverter,
        RayIntSpaceConverter,
        RayBoolSpaceConverter,
    ]

    __type_to_converter = {cls().space_type: cls for cls in __supported_converters}

    @staticmethod
    def get_space_converter(converter_type: str) -> RaySpaceConverter:
        """Return the resource calculator"""
        assert converter_type in RaySpaceConverterFactory.__type_to_converter, f"{converter_type} not supported"
        return RaySpaceConverterFactory.__type_to_converter[converter_type]
