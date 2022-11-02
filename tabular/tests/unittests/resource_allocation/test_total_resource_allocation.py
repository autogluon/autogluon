import pytest
import pandas as pd

from autogluon.core.utils import get_cpu_count, get_gpu_count_all
from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.tabular.models import AbstractModel


class DummyBaseModel(AbstractModel):
    
    def __init__(self, minimum_resources={}, **kwargs):
        self._minimum_resources = minimum_resources
        super().__init__(**kwargs)
        
    def get_minimum_resources(self, **kwargs):
        return self._minimum_resources
    
    def _get_default_resources(self):
        num_cpus = 1
        num_gpus = 1
        return num_cpus, num_gpus


class DummyModel(DummyBaseModel):
    pass

    
class DummyBaggedModel(BaggedEnsembleModel):
    pass


def test_bagged_model_with_total_resources():
    model_base = DummyModel()
    bagged_model = DummyBaggedModel(model_base)
    total_resources = {
        'num_cpus': 1,
        'num_gpus': 0,
    }
    resources = bagged_model._preprocess_fit_resources(total_resources=total_resources)
    assert resources == total_resources
    
    # Given total resources more than what the system has
    total_resources = {
        'num_cpus': 99,
        'num_gpus': 99,
    }
    resources = bagged_model._preprocess_fit_resources(total_resources=total_resources)
    assert resources == {'num_cpus': get_cpu_count(), 'num_gpus': get_gpu_count_all()}
    

def test_bagged_model_with_total_resources_and_ensemble_resources():
    total_resources = {
        'num_cpus': 8,
        'num_gpus': 1,
    }
    model_base = DummyModel()
    bagged_model = DummyBaggedModel(
        model_base,
        hyperparameters={
            'ag_args_fit': {
                'num_cpus': 10,
                'num_gpus': 1,
            }
        }
    )
    with pytest.raises(AssertionError) as e:
        bagged_model._preprocess_fit_resources(total_resources=total_resources)
    
    total_resources = {
        'num_cpus': 8,
        'num_gpus': 1,
    }
    model_base = DummyModel()
    bagged_model = DummyBaggedModel(
        model_base,
        hyperparameters={
            'ag_args_fit': {
                'num_cpus': 4,
                'num_gpus': 1,
            }
        }
    )
    resources = bagged_model._preprocess_fit_resources(total_resources=total_resources)
    assert resources == {'num_cpus': 4, 'num_gpus': 1}
    

def test_bagged_model_without_total_resources():
    model_base = DummyModel()
    bagged_model = DummyBaggedModel(model_base)
    resources = bagged_model._preprocess_fit_resources()
    default_model_num_cpus, default_model_num_gpus = model_base._get_default_resources()
    default_model_resources = {'num_cpus': default_model_num_cpus, 'num_gpus': default_model_num_gpus}
    assert resources == default_model_resources
    

def test_bagged_model_with_total_resources_but_no_gpu_specified():
    model_base = DummyModel()
    total_resources = {
        'num_cpus': 2,
    }
    bagged_model = DummyBaggedModel(model_base)
    resources = bagged_model._preprocess_fit_resources(total_resources=total_resources)
    _, default_model_num_gpus = model_base._get_default_resources()
    default_model_resources = {'num_cpus': 2, 'num_gpus': default_model_num_gpus}
    assert resources == default_model_resources
    
    
def test_bagged_model_without_total_resources_but_with_ensemble_resources():
    model_base = DummyModel()
    bagged_model = DummyBaggedModel(
        model_base,
        hyperparameters={
            'ag_args_fit': {
                'num_cpus': 99,
                'num_gpus': 99,
            }
        }
    )
    with pytest.raises(AssertionError) as e:
        bagged_model._preprocess_fit_resources()
    
    model_base = DummyModel()
    bagged_model = DummyBaggedModel(
        model_base,
        hyperparameters={
            'ag_args_fit': {
                'num_cpus': 1,
                'num_gpus': 0,
            }
        }
    )
    resources = bagged_model._preprocess_fit_resources()
    assert resources == {'num_cpus': 1, 'num_gpus': 0}


def test_nonbagged_model_with_total_resources():
    model_base = DummyModel()
    total_resources = {
        'num_cpus': 1,
        'num_gpus': 0,
    }
    resources = model_base._preprocess_fit_resources(total_resources=total_resources)
    assert resources == total_resources
    
    
def test_nonbagged_model_with_total_resources_but_no_gpu_specified():
    # If model by deafult needs gpu, we use it even if user didn't specify it
    model_base = DummyModel()
    total_resources = {
        'num_cpus': 2,
    }
    resources = model_base._preprocess_fit_resources(total_resources=total_resources)
    _, default_model_num_gpus = model_base._get_default_resources()
    default_model_resources = {'num_cpus': 2, 'num_gpus': default_model_num_gpus}
    assert resources == default_model_resources


def test_nonbagged_model_with_total_resources_and_model_resources():
    model_base = DummyModel(
        hyperparameters={
            'ag_args_fit': {
                'num_cpus': 2,
                'num_gpus': 1
            }
        }
    )
    total_resources = {
        'num_cpus': 1,
        'num_gpus': 1,
    }
    with pytest.raises(AssertionError) as e:
        model_base._preprocess_fit_resources(total_resources=total_resources)
    
    model_base = DummyModel(
        hyperparameters={
            'ag_args_fit': {
                'num_cpus': 1,
                'num_gpus': 1
            }
        }
    )
    total_resources = {
        'num_cpus': 8,
        'num_gpus': 1,
    }
    resources = model_base._preprocess_fit_resources(total_resources=total_resources)
    assert resources == {'num_cpus': 1, 'num_gpus': 1}


def test_nonbagged_model_without_total_resources():
    model_base = DummyModel()
    resources = model_base._preprocess_fit_resources()
    default_model_num_cpus, default_model_num_gpus = model_base._get_default_resources()
    default_model_resources = {'num_cpus': default_model_num_cpus, 'num_gpus': default_model_num_gpus}
    assert resources == default_model_resources
    

def test_nonbagged_model_without_total_resources_but_with_model_resources():
    model_base = DummyModel(
        hyperparameters={
            'ag_args_fit': {
                'num_cpus': 99,
                'num_gpus': 99
            }
        }
    )
    with pytest.raises(AssertionError) as e:
        model_base._preprocess_fit_resources()
    
    model_base = DummyModel(
        hyperparameters={
            'ag_args_fit': {
                'num_cpus': 1,
                'num_gpus': 1
            }
        }
    )
    resources = model_base._preprocess_fit_resources()
    assert resources == {'num_cpus': 1, 'num_gpus': 1}
