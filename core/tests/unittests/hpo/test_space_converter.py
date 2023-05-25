import pytest

from autogluon.common import space

from autogluon.core.hpo.space_converter import RaySpaceConverterFactory
from ray import tune


@pytest.mark.parametrize('space, expected_space',
                        [
                            (space.Categorical([1,2]), tune.choice([1,2])),
                            (space.Real(1,2, log=True), tune.loguniform(1,2)),
                            (space.Real(1,2, log=False), tune.uniform(1,2)),
                            (space.Int(1,2), tune.randint(1,3)),
                            (space.Bool(), tune.randint(0,2)),
                        ])
def test_space_converter(space, expected_space):
    ray_space = RaySpaceConverterFactory.get_space_converter(space.__class__.__name__).convert(space)
    assert type(ray_space) == type(expected_space)
