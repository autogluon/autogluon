from autogluon.multimodal.utils.registry import Registry


def test_registry():
    MODEL_REGISTRY = Registry("MODEL")

    @MODEL_REGISTRY.register()
    class MyModel:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    @MODEL_REGISTRY.register()
    def my_model():
        return

    @MODEL_REGISTRY.register("test_class")
    class MyModelWithNickName:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

    @MODEL_REGISTRY.register("test_function")
    def my_model_with_nick_name():
        return

    class MyModel2:
        pass

    MODEL_REGISTRY.register(MyModel2)
    MODEL_REGISTRY.register("my_model2", MyModel2)
    assert MODEL_REGISTRY.list_keys() == [
        "MyModel",
        "my_model",
        "test_class",
        "test_function",
        "MyModel2",
        "my_model2",
    ]
    model = MODEL_REGISTRY.create("MyModel", 1, 2)
    assert model.a == 1 and model.b == 2
    model = MODEL_REGISTRY.create("MyModel", a=2, b=3)
    assert model.a == 2 and model.b == 3
    model = MODEL_REGISTRY.create_with_json("MyModel", "[4, 5]")
    assert model.a == 4 and model.b == 5
    model = MODEL_REGISTRY.create_with_json("test_class", '{"a": 100, "b": 200, "c": 300}')
    assert model.a == 100 and model.b == 200 and model.c == 300
    assert MODEL_REGISTRY.get("test_class") == MyModelWithNickName
