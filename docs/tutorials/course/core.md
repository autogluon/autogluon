# Search Space and Decorator

This tutorial explains the supported search spaces and how to use them, including
simple search spaces (:class:`autogluon.space.Int`, :class:`autogluon.space.Real`,
:class:`autogluon.space.Categorical`) and nested search spaces
(:class:`autogluon.space.Categorical`, :class:`autogluon.space.List`,
:class:`autogluon.space.Dict`).
AutoGluon also enables search spaces in user-defined objects using the decorator
:func:`autogluon.obj` and user-defined functions using the decorator
:func:`autogluon.func`.

## Search Space

### Simple Search Space

```{.python .input}
import autogluon as ag
```

- Integer Space :class:`autogluon.space.Int`

An integer will be chosen between lower and upper value during the
searcher sampleing.

```{.python .input}
a = ag.space.Int(lower=0, upper=10)
print(a)
```

- Real Space :class:`autogluon.space.Real`

An real number will be chosen between lower and upper value during the
searcher sampleing.

```{.python .input}
b = ag.space.Real(lower=1e-4, upper=1e-2)
print(b)
```

Real space in log scale:

```{.python .input}
c = ag.space.Real(lower=1e-4, upper=1e-2, log=True)
print(c)
```

- Categorical Space :class:`autogluon.space.Categorical`

Categorical Space will chooce one choice from all the possible values during
the searcher sampling.

```{.python .input}
d = ag.space.Categorical('Monday', 'Tuesday', 'Wednesday')
print(d)
```

### Nested Search Space

- Categorical Space :class:`autogluon.space.Categorical`

Categorical Space can also be used as a nested search space.

```{.python .input}
e = ag.space.Categorical(
        'densenet269',
        ag.space.Categorical('resnet50', 'resnet101'),
    )
print(e)
```

- List Space :class:`autogluon.space.List`

List Space returns a list of sampled results.

```{.python .input}
f = ag.space.List(
        ag.space.Int(0, 3),
        ag.space.Categorical('alpha', 'beta'),
    )
print(f)
```

- Dict Space :class:`autogluon.space.List`

Dict Space returns a dict of sampled results.

```{.python .input}
g = ag.space.Dict(
        key1=ag.space.Categorical('alpha', 'beta'),
        key2=ag.space.Int(0, 3),
    )
print(g)
```

## Decorators for Searchbale Object and Customized Training Scripts

- Searchable space in customized class :func:`autogluon.obj`

```{.python .input}
@ag.obj(
    name=ag.space.Categorical('auto', 'gluon'),
)
class myobj:
    def __init__(self, name):
        self.name = name
h = myobj()
print(h)
```

- Searchable space in customized function :func:`autogluon.obj`

```{.python .input}
@ag.func(
    framework=ag.space.Categorical('mxnet', 'pytorch'),
)
def myfunc(framework):
    return framework
i = myfunc()
print(i)
```

We can also make them inside a nested space:

```{.python .input}
j =ag.space.Dict(
        a=ag.Real(0, 10),
        obj1=myobj(),
        obj2=myfunc(),
    ),
print(j)
```

- Customized Train Script using :func:`autogluon.args`

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
            obj=myobj(),
        ),
    h=ag.space.Categorical('test', myobj()),
    i = myfunc(),
)
def train_fn(args, reporter):
    print('args: {}'.format(args))
```

## Create Searcher and Sample A Configuration

- Create a searcher and sample configuration.

```{.python .input}
searcher = ag.searcher.RandomSearcher(train_fn.cs)
config = searcher.get_config()
print(config)
```

- Run one training job with the sampled configuration:

```{.python .input}
train_fn(train_fn.args, config)
```

Exit AutoGluon:

```{.python .input}
ag.done()
```
