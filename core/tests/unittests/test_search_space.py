import autogluon.core as ag


def test_search_space():
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
            obj=ag.space.Categorical('auto', 'gluon'),
        ),
        h=ag.space.Categorical('test', ag.space.Categorical('auto', 'gluon')),
        i=ag.space.Categorical('mxnet', 'pytorch'),
    )
    def train_fn(args, reporter):
        a, b, c, d, e, f, g, h, i = args.a, args.b, args.c, args.d, args.e, \
                                    args.f, args.g, args.h, args.i

        class myobj:
            def __init__(self, name):
                self.name = name

        def myfunc(framework):
            return framework

        assert a <= 1e-2 and a >= 1e-3
        assert b <= 1e-2 and b >= 1e-3
        assert c <= 10 and c >= 1
        assert d in ['a', 'b', 'c', 'd']
        assert e in [True, False]
        assert f[0] in [1, 2]
        assert f[1] in [4, 5]
        assert g['a'] <= 10 and g['a'] >= 0
        assert myobj(g.obj).name in ['auto', 'gluon']
        assert e in [True, False]
        assert h in ['test', 'auto', 'gluon']
        assert myfunc(i) in ['mxnet', 'pytorch']
        reporter(epoch=1, accuracy=0)

    scheduler = ag.scheduler.FIFOScheduler(train_fn,
                                           resource={'num_cpus': 4, 'num_gpus': 0},
                                           num_trials=10,
                                           reward_attr='accuracy',
                                           time_attr='epoch',
                                           checkpoint=None)
    scheduler.run()
    scheduler.join_jobs()


def test_search_space_dot_key():
    @ag.args(
        **{'model.name': ag.space.Categorical('mxnet', 'pytorch')}
    )
    def train_fn(args, reporter):
        assert args['model.name'] == 'mxnet' or args['model.name'] == 'pytorch'

    scheduler = ag.scheduler.FIFOScheduler(train_fn, num_trials=2)
    scheduler.run()
    scheduler.join_jobs()
