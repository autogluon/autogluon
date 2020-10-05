# Search Space and Decorator

This tutorial explains the supported search spaces and how to use them, including
simple search spaces (Int, Real, and Categorical) and nested search spaces
(Categorical, List, Dict). Each search space describes the set of possible values for a hyperparameter, from which the searcher will try particular values during hyperparameter optimization. AutoGluon also enables search spaces in user-defined objects using the decorator
`ag.obj` and user-defined functions using the decorator `ag.func`.

## Search Space

### Simple Search Space

```{.python .input}
import autogluon.core as ag
```

#### Integer Space :class:`autogluon.core.space.Int`

An integer is chosen between lower and upper value during the
searcher sampling.

```{.python .input}
a = ag.space.Int(lower=0, upper=10)
print(a)
```

Get default value:

```{.python .input}
a.default
```

Change default value, which is the first configuration that a random searcher
:class:`autogluon.core.searcher.RandomSearcher` will try:

```{.python .input}
a = ag.space.Int(lower=0, upper=10, default=2)
print(a.default)
```

Pick a random value.

```{.python .input}
a.rand
```

#### Real Space :class:`autogluon.core.space.Real`

A real number is chosen between lower and upper value during the
searcher sampling.

```{.python .input}
b = ag.space.Real(lower=1e-4, upper=1e-2)
print(b)
```

Real space in log scale:

```{.python .input}
c = ag.space.Real(lower=1e-4, upper=1e-2, log=True)
print(c)
```

#### Categorical Space :class:`autogluon.core.space.Categorical`

Categorical Space chooses one value from all the possible values during
the searcher sampling.

```{.python .input}
d = ag.space.Categorical('Monday', 'Tuesday', 'Wednesday')
print(d)
```

### Nested Search Space

#### Categorical Space :class:`autogluon.core.space.Categorical`

Categorical Space can also be used as a nested search space.
For an example, see NestedExampleObj_.


#### List Space :class:`autogluon.core.space.List`

List Space returns a list of sampled results.

In this example, the first element of the list is a Int Space sampled
from 0 to 3, and the second element is a Categorical Space sampled
from the choices of `'alpha'` and `'beta'`.

```{.python .input}
f = ag.space.List(
        ag.space.Int(0, 3),
        ag.space.Categorical('alpha', 'beta'),
    )
print(f)
```

Get one example configuration:

```{.python .input}
f.rand
```

#### Dict Space :class:`autogluon.core.space.Dict`

Dict Space returns a dict of sampled results.

Similar to List Space, the resulting configuraton of Dict is
a dict. In this example, the value of `'key1'` is sampled from
a Categorical Space with the choices of `'alpha'` and `'beta'`,
and the value of `'key2'` is sampled from an Int Space between
0 and 3.

```{.python .input}
g = ag.space.Dict(
        key1=ag.space.Categorical('alpha', 'beta'),
        key2=ag.space.Int(0, 3),
        key3='constant'
    )
print(g)
```

Get one example configuration:

```{.python .input}
g.rand
```

### Decorators for Searchbale Object and Customized Training Scripts

In this section, we show how to insert search space into customized objects and
training functions.

#### Searchable Space in Customized Class :func:`autogluon.obj`

In AutoGluon searchable object can be returned by a user defined class with a decorator.

```{.python .input}
@ag.obj(

    name=ag.space.Categorical('auto', 'gluon'),
    static_value=10,
    rank=ag.space.Int(2, 5),
)
class MyObj:
    def __init__(self, name, rank, static_value):
        self.name = name
        self.rank = rank
        self.static_value = static_value
    def __repr__(self):
        repr = 'MyObj -- name: {}, rank: {}, static_value: {}'.format(
                self.name, self.rank, self.static_value)
        return repr
h = MyObj()
print(h)
```

Get one example random object:

```{.python .input}
h.rand
```

.. _NestedExampleObj:

We can also use it within a Nested Space such as :class:`autogluon.core.space.Categorical`.
In this example, the resulting nested space will be sampled from: 

```{.python .input}
nested = ag.space.Categorical(
        ag.space.Dict(
                obj1='1',
                obj2=ag.space.Categorical('a', 'b'),
            ),
        MyObj(),
    )

print(nested)
```

Get an example output:

```{.python .input}
for _ in range(5):
    result = nested.rand
    assert (isinstance(result, dict) and result['obj2'] in ['a', 'b']) or hasattr(result, 'name')
    print(result)
```

#### Searchable Space in Customized Function :func:`autogluon.obj`

We can also insert a searchable space in a customized function:

```{.python .input}
@ag.func(
    framework=ag.space.Categorical('mxnet', 'pytorch'),
)
def myfunc(framework):
    return framework
i = myfunc()
print(i)
```

We can also put a searchable space inside a nested space:

```{.python .input}
j = ag.space.Dict(
        a=ag.Real(0, 10),
        obj1=MyObj(),
        obj2=myfunc(),
    )
print(j)
```

#### Customized Train Script Using :func:`autogluon.args`

`train_func` is where to put your model training script, which takes in various keyword `args` as its hyperparameters and reports the performance of the trained model using the provided `reporter`. Here, we show a dummy train_func that simply prints these objects.

```{.python .input}
@ag.args(
    a=ag.space.Int(1, 10),
    b=ag.space.Real(1e-3, 1e-2),
    c=ag.space.Real(1e-3, 1e-2, log=True),
    d=ag.space.Categorical('a', 'b', 'c', 'd'),
    e=ag.space.Bool(),
    f=ag.space.List(
            ag.space.Int(1, 2),
            ag.space.Categorical(4, 5),
        ),
    g=ag.space.Dict(
            a=ag.Real(0, 10),
            obj=MyObj(),
        ),
    h=ag.space.Categorical('test', MyObj()),
    i = myfunc(),
)
def train_fn(args, reporter):
    print('args: {}'.format(args))
```

## Create Searcher and Sample a Configuration

In this section, we create a Searcher object, which orchestrates a particular hyperparameter-tuning strategy.

#### Create a Searcher and Sample Configuration

```{.python .input}
searcher = ag.searcher.RandomSearcher(train_fn.cs)
config = searcher.get_config()
print(config)
```

#### Run one training job with the sampled configuration:

```{.python .input}
train_fn(train_fn.args, config)
```
