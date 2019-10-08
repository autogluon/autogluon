import copy
import logging
import argparse
import functools
import collections
import numpy as np
import multiprocessing as mp
import ConfigSpace as CS

from ..searcher import Sample
from .space import *
from ..utils import EasyDict as ezdict
from ..utils.deprecate import make_deprecate

__all__ = ['autogluon_method', 'autogluon_kwargs', 'autogluon_object',
           'autogluon_function', 'autogluon_register_args',
           'autogluon_register_dict']

logger = logging.getLogger(__name__)

class autogluon_method(object):
    SEED = mp.Value('i', 0)
    LOCK = mp.Lock()
    def __init__(self, f):
        functools.update_wrapper(self, f)
        self.f = f

    def __call__(self, args, config, **kwargs):
        args = copy.deepcopy(args)
        new_config = copy.deepcopy(config)
        self._rand_seed()
        striped_keys = [k.split('.')[0] for k in new_config.keys()]
        
        if isinstance(args, argparse.Namespace) or isinstance(args, argparse.ArgumentParser):
            args_dict = vars(args)
        else:
            args_dict = args

        for k, v in args_dict.items():
            # handle different type of configurations
            if k in striped_keys:
                if isinstance(v, Sequence):
                    sub_config = strip_config_space(new_config, prefix=k)
                    args_dict[k] = []
                    for idx, obj in enumerate(v):
                        min_config = strip_config_space(sub_config, prefix=str(idx))
                        assert isinstance(obj, AutoGluonObject)
                        args_dict[k].append(obj._lazy_init(**min_config))
                elif isinstance(v, AutoGluonObject):
                    sub_config = strip_config_space(new_config, prefix=k)
                    args_dict[k] = v._lazy_init(**sub_config)
                elif isinstance(v, Choice):
                    sub_config = strip_config_space(new_config, prefix=k)
                    choice = sub_config.pop(k)
                    if isinstance(v[choice], AutoGluonObject):
                        # nested space: Choice of AutoGluonobjects
                        min_config = strip_config_space(sub_config, prefix=str(choice))
                        args_dict[k] = v[choice]._lazy_init(**min_config)
                    else:
                        args_dict[k] = v[choice]
                elif isinstance(new_config[k], Sample):
                    args_dict[k] = new_config[k]()
                else:
                    if '.' in k:
                        continue
                    args_dict[k] = new_config[k]
            elif isinstance(v, AutoGluonObject):
                args_dict[k] = v._lazy_init()
                

        output = self.f(args, **kwargs)
        if 'reporter' in kwargs and kwargs['reporter'] is not None:
            logger.debug('Reporter Done!')
            kwargs['reporter'](done=True)
        return output
 
    def _register_args(self, default, **kwvars):
        self.cs = CS.ConfigurationSpace()
        self.args = ezdict()
        self.args.update(default)
        self.update(**kwvars)

    def update(self, **kwargs):
        """For searcher support ConfigSpace
        """
        self.kwvars = kwargs
        for k, v in kwargs.items():
            if isinstance(v, (Choice, Sequence)):
                sub_cs = v.get_config_space(k)
                _add_cs(self.cs, sub_cs, k)
                self.args.update({k: v})
            elif isinstance(v, Space):
                hp = v.get_config_space(name=k)
                _add_hp(self.cs, hp)
                self.args.update({k: hp.default_value})
            elif isinstance(v, AutoGluonObject):
                _add_cs(self.cs, v.cs, k)
                self.args.update({k: v})
            else:
                _rm_hp(self.cs, k)
                self.args.update({k: v})

    def get_kwspaces(self):
        """For RL searcher/controller
        """
        self.kwspaces = collections.OrderedDict()
        for k, v in self.kwvars.items():
            if isinstance(v, Sequence):
                for idx, obj in enumerate(v):
                    assert isinstance(obj, AutoGluonObject)
                    for sub_k, sub_v in obj.kwspaces.items():
                        new_k = '{}.{}.{}'.format(k, idx, sub_k)
                        self.kwspaces[new_k] = sub_v
            elif isinstance(v, Choice):
                new_k = '{}.{}'.format(k, k)
                self.kwspaces[new_k] = v
                for idx, obj in enumerate(v):
                    if isinstance(obj, AutoGluonObject):
                        for idx, (sub_k, sub_v) in enumerate(obj.kwspaces.items()):
                            new_k = '{}.{}.{}'.format(k, idx, sub_k)
                            self.kwspaces[new_k] = sub_v
            elif isinstance(v, AutoGluonObject):
                for sub_k, sub_v in v.kwspaces.items():
                    new_k = '{}.{}'.format(k, sub_k)
                    self.kwspaces[new_k] = sub_v
            elif isinstance(v, Space):
                self.kwspaces[k] = v
        return self.kwspaces

    def _rand_seed(self):
        autogluon_method.SEED.value += 1
        np.random.seed(autogluon_method.SEED.value)

    def __repr__(self):
        return repr(self.f)

def _add_hp(cs, hp):
    if hp.name in cs._hyperparameters:
        cs._hyperparameters[hp.name] = hp
    else:
        cs.add_hyperparameter(hp)

def _add_cs(master_cs, sub_cs, prefix, delimiter='.', parent_hp=None):
    new_parameters = []
    for hp in sub_cs.get_hyperparameters():
        new_parameter = copy.copy(hp)
        # Allow for an empty top-level parameter
        if new_parameter.name == '':
            new_parameter.name = prefix
        elif not prefix == '':
            new_parameter.name = "%s%s%s" % (prefix, '.',
                                             new_parameter.name)
        new_parameters.append(new_parameter)
    #if len(new_parameters) > 0 and new_parameters[0].name not in master_cs._hyperparameters:
    #    master_cs.add_configuration_space(prefix, sub_cs, delimiter, parent_hyperparameter=parent_hp)
    #else:
    #    for hp in new_parameters:
    #        master_cs._hyperparameters[hp.name] = hp
    for hp in new_parameters:
        _add_hp(master_cs, hp)

def _rm_hp(cs, k):
    if k in cs._hyperparameters:
        cs._hyperparameters.pop(k)
    for hp in cs.get_hyperparameters():
        if  hp.name.startswith("%s."%(k)):
            cs._hyperparameters.pop(hp.name)

def autogluon_register_args(default={}, **kwvars):
    """Decorator for customized training script, registering arguments or searchable spaces
    to the decorated function. The arguments should be python objects, autogluon objects
    (see :function:`autogluon.autogluon_object`_ .), or autogluon search spaces
    (:class:`autogluon.Int`, :class:`autogluon.Linear` ...).

    Example:
        >>> @autogluon_register_args(batch_size=10, lr=ag.Linear(0.01, 0.1))
        >>> def my_train(args):
        >>>     print('Batch size is {}, LR is {}'.format(args.batch_size, arg.lr))

    """
    kwvars['default_config'] = default
    def registered_func(func):
        @autogluon_method
        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            return func(*args, **kwargs)

        default = kwvars['default_config']
        if isinstance(default, argparse.Namespace) or isinstance(default, argparse.ArgumentParser):
            default = vars(default)
        wrapper_call._register_args(default=default, **kwvars)
        return wrapper_call

    return registered_func

def autogluon_kwargs(**kwvars):
    def registered_func(func):
        cs = CS.ConfigurationSpace()
        kwspaces = collections.OrderedDict()
        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            kwvars.update(kwargs)
            for k, v in kwvars.items():
                if isinstance(v, Choice):
                    kwargs[k] = v
                    kwspaces[k] = v
                    sub_cs = v.get_config_space(name=k)
                    _add_cs(cs, sub_cs, '', '')
                elif isinstance(v, Space):
                    kwspaces[k] = v
                    hp = v.get_config_space(name=k)
                    _add_hp(cs, hp)
                    kwargs[k] = hp.default_value
                    #else:
                    #    kwargs[k] = cs.get_hyperparameter(k).default_value
                else:
                    _rm_hp(cs, k)
                    kwargs[k] = v
            return func(*args, **kwargs)
        wrapper_call.cs = cs
        wrapper_call.kwspaces = kwspaces
        return wrapper_call
    return registered_func

def autogluon_function(**kwvars):
    """Register args or searchable spaces to the functions.

    Return:
        AutoGluonobject: a lazy init object, which allows distributed training.

    Example:
        >>> from gluoncv.model_zoo import get_model
        >>> 
        >>> @ag.autogluon_function(pretrained=ag.Choice(True, False))
        >>> def cifar_resnet(pretrained):
        >>>     return get_model('cifar_resnet20_v1', pretrained=pretrained)
    """
    def registered_func(func):
        class autogluonobject(AutoGluonObject):
            @autogluon_kwargs(**kwvars)
            def __init__(self, *args, **kwargs):
                self.func = func
                self.args = args
                self.kwargs = kwargs
                self._inited = False

            def __call__(self, *args, **kwargs):
                if not self._inited:
                    self._inited = True
                    config = self.cs.sample_configuration().get_dictionary()
                    self._lazy_init(**config)
                return self._instance.__call__(*args, **kwargs)

            def _lazy_init(self, **nkwvars):
                # lazy initialization for passing config files
                self.kwargs.update(nkwvars)
                for k, v in self.kwargs.items():
                    if k in self.kwspaces and isinstance(self.kwspaces[k], Choice):
                        self.kwargs[k] = self.kwspaces[k][v]
                        
                self._instance = self.func(*self.args, **self.kwargs)
                return self._instance

            def __repr__(self):
                return 'AutoGluonObject'

        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            agobj = autogluonobject(*args, **kwargs)
            agobj.cs = agobj.__init__.cs
            agobj.kwspaces = agobj.__init__.kwspaces
            return agobj
        return wrapper_call
    return registered_func

def autogluon_object(**kwvars):
    """Register args or searchable spaces to the class.

    Return:
        AutoGluonobject: a lazy init object, which allows distributed training.

    Example:
        >>> from gluoncv.model_zoo.cifarresnet import CIFARResNetV1, CIFARBasicBlockV1
        >>>
        >>> @autogluon_object(
        >>>     learning_rate=ag.LogLinear(1e-4, 1e-1),
        >>>     wd=ag.Linear(1e-4, 1e-1),
        >>> )
        >>> class Adam(optim.Adam):
        >>>     pass

    """
    def registered_class(Cls):
        class autogluonobject(AutoGluonObject):#, Cls
            @autogluon_kwargs(**kwvars)
            def __init__(self, *args, **kwargs):
                self._args = args
                self._kwargs = kwargs
                self._inited = False

            def __call__(self, *args, **kwargs):
                if not self._inited:
                    self._inited = True
                    config = autogluonobject.cs.sample_configuration().get_dictionary()
                    self._lazy_init(**config)
                return self.__call__(*args, **kwargs)

            def init(self):
                return self._lazy_init()

            def _lazy_init(self, **nkwvars):
                kwargs = self._kwargs
                kwargs.update(nkwvars)
                for k, v in kwargs.items():
                    if k in autogluonobject.kwspaces and isinstance(autogluonobject.kwspaces[k], Choice):
                        kwargs[k] = autogluonobject.kwspaces[k][v]
                args = self._args
                return Cls(*args, **kwargs)

            def __repr__(self):
                return 'AutoGluonObject -- ' + Cls.__name__

        autogluonobject.cs = autogluonobject.__init__.cs
        autogluonobject.kwspaces = autogluonobject.__init__.kwspaces
        return autogluonobject

    return registered_class

autogluon_register_dict = make_deprecate(autogluon_register_args, 'autogluon_register_dict')
