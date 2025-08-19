import pytest

from autogluon.timeseries.models.registry import ModelRegistry


@pytest.fixture()
def register_classes():
    registry_backup = ModelRegistry.REGISTRY
    ModelRegistry.REGISTRY = {}

    class FooModel(metaclass=ModelRegistry):
        pass

    class BarModel(metaclass=ModelRegistry):
        pass

    yield FooModel, BarModel

    ModelRegistry.REGISTRY = registry_backup


def test_when_models_initialized_then_models_are_registered(register_classes):
    for k in ["Bar", "Foo"]:
        assert k in ModelRegistry.REGISTRY.keys()


def test_when_models_initialized_without_priority_then_model_priority_is_zero(register_classes):
    assert ModelRegistry.get_model_priority("Foo") == 0
    assert ModelRegistry.get_model_priority("Bar") == 0


def test_when_models_initialized_then_model_class_is_given(register_classes):
    assert ModelRegistry.get_model_class("Foo") is register_classes[0]
    assert ModelRegistry.get_model_class("Bar") is register_classes[1]


def test_when_models_registered_with_no_model_suffix_then_it_is_registered(register_classes):
    class Baz(metaclass=ModelRegistry):
        pass

    assert "Baz" in ModelRegistry.REGISTRY.keys()


def test_when_models_registered_with_priority_then_priority_is_correct(register_classes):
    class BazModel(metaclass=ModelRegistry):
        ag_priority = 10
        pass

    assert ModelRegistry.get_model_priority("Baz") == 10


def test_when_models_registered_with_aliases_then_aliases_registered(register_classes):
    class BazModel(metaclass=ModelRegistry):
        ag_model_aliases = ["Qux"]
        pass

    assert ModelRegistry.get_model_class("Qux") is BazModel


def test_when_multiple_models_with_same_alias_registered_then_value_error_raised(register_classes):
    with pytest.raises(ValueError, match="model already exists"):

        class BazModel(metaclass=ModelRegistry):
            ag_model_aliases = ["Qux"]
            pass

        class QuxModel(metaclass=ModelRegistry):
            pass


def test_when_unknown_model_requested_then_value_error_raised(register_classes):
    with pytest.raises(ValueError, match="Unknown model:"):
        ModelRegistry.get_model_class("Unknown")
