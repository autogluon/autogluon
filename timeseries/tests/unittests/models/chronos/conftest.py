import pytest

# TODO: this is a temporary measure to point to a test fixture on Hugging Face. Once
# models are released, this will be removed.
CHRONOS_BOLT_TEST_MODEL_PATH = "autogluon/chronos-bolt-350k-test"


@pytest.fixture(scope="module", params=["default", "bolt-t5-efficient-350k"])
def chronos_model_path(request, hf_model_path):
    if request.param == "default":
        yield hf_model_path
    elif request.param == "bolt-t5-efficient-350k":
        yield CHRONOS_BOLT_TEST_MODEL_PATH
