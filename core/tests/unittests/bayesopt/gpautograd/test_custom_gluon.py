from autograd import numpy as np

from autogluon.core.searcher.bayesopt.gpautograd.gluon import Parameter, \
    ParameterDict, Block


def test_parameter():
    p = Parameter(name='abc', shape=(1,))
    p.initialize()
    data = p.data
    grad = p.grad


def test_parameter_dict():
    pd = ParameterDict('pd')
    pd.initialize()
    p = pd.get('def')


def test_block():
    class TestBlock(Block):
        def __init__(self):
            super(TestBlock, self).__init__()
            with self.name_scope():
                self.a = self.params.get('a', shape=(10,))
                self.b = self.params.get('b', shape=(10, ))

        def forward(self, x):
            return x + self.a.data() + self.b.data()

    t = TestBlock()
    t.initialize()
    print(t.a.grad_req)
    t.a.set_data(np.ones((10, )))
    assert 'a' in t.a.name
    x = np.zeros((10, ))
    y = t(x)
    print(y)
