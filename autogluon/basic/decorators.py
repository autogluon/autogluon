import copy
import logging
import argparse
import functools
import numpy as np
import multiprocessing as mp
import ConfigSpace as CS

from ..searcher import Sample
from .space import *
from ..utils import EasyDict as ezdict
from .deprecate import _make_deprecate

__all__ = ['autogluon_method', 'autogluon_kwargs', 'autogluon_object',
           'autogluon_function', 'autogluon_register_args',
           'autogluon_register_dict']

logger = logging.getLogger(__name__)

class autogluon_method(object):
    """Enable searcher to update default function args.
    """
    SEED = mp.Value('i', 0)
    LOCK = mp.Lock()
    def __init__(self, f):
        functools.update_wrapper(self, f)
        self.f = f

    def __call__(self, args, config, **kwargs):
        #print('args', args)
        args = copy.deepcopy(args)
        new_config = copy.deepcopy(config)
        self._rand_seed()
        striped_keys = [k.split('.')[0] for k in new_config.keys()]
        
        if isinstance(args, argparse.Namespace) or isinstance(args, argparse.ArgumentParser):
            #print('\n\n Using Namespace')
            args_dict = vars(args)
        else:
            args_dict = args

        for k, v in args_dict.items():
            # handle different type of configurations
            if k in striped_keys:
                if isinstance(v, AutoGluonObject):
                    sub_config = strip_cofing_space(new_config, prefix=k)
                    args_dict[k] = v._lazy_init(**sub_config)
                elif isinstance(v, ListSpace):
                    sub_config = strip_cofing_space(new_config, prefix=k)
                    choice = sub_config.pop(k)
                    if isinstance(v[choice], AutoGluonObject):
                        # nested space: List of AutoGluonobjects
                        sub_config = strip_cofing_space(sub_config, prefix=str(choice))
                        args_dict[k] = v[choice]._lazy_init(**sub_config)
                    else:
                        args_dict[k] = v[choice]
                elif isinstance(new_config[k], Sample):
                    args_dict[k] = new_config[k]()
                else:
                    if '.' in k:
                        continue
                    args_dict[k] = new_config[k]

        self.f(args, **kwargs)
        if 'reporter' in kwargs and kwargs['reporter'] is not None:
            logger.debug('Reporter Done!')
            kwargs['reporter'](done=True)
 
    def _register_args(self, default, **kwvars):
        self.cs = CS.ConfigurationSpace()
        self.args = ezdict()
        self.args.update(default)
        self.update(**kwvars)

    def update(self, **kwargs):
        self.kwvars = kwargs
        for k, v in kwargs.items():
            if isinstance(v, ListSpace):
                sub_cs = v.get_config_space(name=k)
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
        else:
            new_parameter.name = "%s%s%s" % (prefix, '.',
                                             new_parameter.name)
        new_parameters.append(new_parameter)
    if new_parameters[0].name not in master_cs._hyperparameters:
        master_cs.add_configuration_space(prefix, sub_cs, delimiter, parent_hyperparameter=parent_hp)
    else:
        for hp in new_parameters:
            master_cs._hyperparameters[hp.name] = hp

def _rm_hp(cs, k):
    if k in cs._hyperparameters:
        cs._hyperparameters.pop(k)
    for hp in cs.get_hyperparameters():
        if "%s."%(k) in hp.name:
            cs._hyperparameters.pop(hp.name)

def autogluon_register_args(default={}, **kwvars):
    """Register default args or searchable spaces to the 'autogluon_method'
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
    """Decorating function and gather configspaces
    """
    def registered_func(func):
        cs = CS.ConfigurationSpace()
        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            kwvars.update(kwargs)
            for k, v in kwvars.items():
                if isinstance(v, ListSpace):
                    kwargs[k] = v
                    if k not in cs.get_hyperparameter_names():
                        sub_cs = v.get_config_space(name=k)
                        _add_cs(cs, sub_cs, '', '')
                elif isinstance(v, Space):
                    #if k not in cs.get_hyperparameter_names():
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
        return wrapper_call
    return registered_func

def autogluon_function(**kwvars):
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

            def _lazy_init(self, **kwvars):
                # lazy initialization for passing config files
                self.kwargs.update(kwvars)
                self._instance = self.func(*self.args, **self.kwargs)
                return self._instance

        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            agobj = autogluonobject(*args, **kwargs)
            agobj.cs = agobj.__init__.cs
            return agobj
        return wrapper_call
    return registered_func

def autogluon_object(**kwvars):
    """Register args or searchable spaces to the class.
    AutoGluon object is a lazy init object, which allows distributed training.
    """
    def registered_class(Cls):
        class autogluonobject(AutoGluonObject, Cls):
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

            def _lazy_init(self, **kwvars):
                self.__class__ = Cls
                kwargs = self._kwargs
                kwargs.update(kwvars)
                args = self._args
                del self._args
                del self._kwargs
                self.__init__(*args, **kwargs)

                return self

            def __repr__(self):
                return Cls.__repr__(self)

        autogluonobject.cs = autogluonobject.__init__.cs
        return autogluonobject

    return registered_class

autogluon_register_dict = _make_deprecate(autogluon_register_args, 'autogluon_register_dict')
