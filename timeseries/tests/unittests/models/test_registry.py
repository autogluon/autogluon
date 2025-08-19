import pytest

from autogluon.timeseries.models.registry import ModelRegistry


@pytest.fixture()
def register_classes():
    class FooModel(metaclass=ModelRegistry):
        pass

    class BarModel(metaclass=ModelRegistry):
        pass

    yield FooModel, BarModel

    ModelRegistry.REGISTRY.pop("Foo", None)
    ModelRegistry.REGISTRY.pop("Bar", None)


def test_when_models_initialized_then_models_are_registered(register_classes):
    for k in ["Bar", "Foo"]:
        assert k in ModelRegistry.REGISTRY.keys()


def test_when_models_initialized_without_default_priority_then_model_priority_is_zero(register_classes):
    assert ModelRegistry.get_model_priority("Foo") == 0
    assert ModelRegistry.get_model_priority("Bar") == 0


def test_when_models_initialized_then_model_class_is_given(register_classes):
    assert ModelRegistry.get_model_class("Foo") is register_classes[0]
    assert ModelRegistry.get_model_class("Bar") is register_classes[1]


def test_when_models_registered_with_no_model_suffix_then_it_is_registered():
    try:

        class Baz(metaclass=ModelRegistry):
            pass

        assert "Baz" in ModelRegistry.REGISTRY.keys()
    finally:
        ModelRegistry.REGISTRY.pop("Baz", None)


def test_when_models_registered_with_default_priority_then_priority_is_correct():
    try:

        class BazModel(metaclass=ModelRegistry):
            default_priority = 10
            pass

        assert ModelRegistry.get_model_priority("Baz") == 10
    finally:
        ModelRegistry.REGISTRY.pop("Baz", None)


def test_when_models_registered_with_aliases_then_aliases_registered():
    try:

        class BazModel(metaclass=ModelRegistry):
            _aliases = ["Qux"]
            pass

        assert ModelRegistry.get_model_class("Qux") is BazModel
    finally:
        ModelRegistry.REGISTRY.pop("Baz", None)
        ModelRegistry.REGISTRY.pop("Qux", None)


def test_when_multiple_models_with_same_alias_registered_then_value_error_raised():
    try:

        class BazModel(metaclass=ModelRegistry):
            _aliases = ["Qux"]
            pass

        class QuxModel(metaclass=ModelRegistry):
            pass
    except ValueError:
        pass
    finally:
        ModelRegistry.REGISTRY.pop("Baz", None)
        ModelRegistry.REGISTRY.pop("Qux", None)
