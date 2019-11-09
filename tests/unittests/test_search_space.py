import logging
import numpy as np
import autogluon as ag

@ag.obj(
    name=ag.space.Categorical('auto', 'gluon'),
)
class myobj:
    def __init__(self, name):
        self.name = name

@ag.func(
    framework=ag.space.Categorical('mxnet', 'pytorch'),
)
def myfunc(framework):
    return framework

@ag.args(
    a=ag.space.Real(1e-3, 1e-2, log=True),
    b=ag.space.Real(1e-3, 1e-2),
    c=ag.space.Int(1, 10),
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
    a, b, c, d, e, f, g, h, i = args.a, args.b, args.c, args.d, args.e, \
            args.f, args.g, args.h, args.i
    assert a <= 1e-2 and a >= 1e-3
    assert b <= 1e-2 and b >= 1e-3
    assert c <= 10 and c >= 1
    assert d in ['a', 'b', 'c', 'd']
    assert e in [True, False]
    assert f[0] in [1, 2]
    assert f[1] in [4, 5]
    assert g['a'] <= 10 and g['a'] >= 0
    assert g.obj.name in ['auto', 'gluon']
    assert hasattr(h, 'name') or h == 'test'
    assert i in ['mxnet', 'pytorch']
    reporter(epoch=e, accuracy=0)

def test_fifo_scheduler():
    scheduler = ag.scheduler.FIFOScheduler(train_fn,
                                           resource={'num_cpus': 2, 'num_gpus': 0},
                                           num_trials=20,
                                           reward_attr='accuracy',
                                           time_attr='epoch')
    scheduler.run()
    scheduler.join_jobs()

if __name__ == '__main__':
    import nose
    nose.runmodule()
    ag.done()
