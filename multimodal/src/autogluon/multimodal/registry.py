import json
from json import JSONDecodeError
from typing import List


class Registry:
    """
    Create the registry that will map name to object.
    This facilitates the users to create custom registry.
    """

    def __init__(self, name: str) -> None:
        """
        Parameters
        ----------
        name
            Registry name
        """
        self._name: str = name
        self._obj_map: dict[str, object] = dict()

    def _do_register(self, name: str, obj: object) -> None:
        assert name not in self._obj_map, "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, *args):
        """
        Register the given object under either the nickname or `obj.__name__`. It can be used as
         either a decorator or not. See docstring of this class for usage.
        """
        if len(args) == 2:
            # Register an object with nick name by function call
            nickname, obj = args
            self._do_register(nickname, obj)
        elif len(args) == 1:
            if isinstance(args[0], str):
                # Register an object with nick name by decorator
                nickname = args[0]

                def deco(func_or_class: object) -> object:
                    self._do_register(nickname, func_or_class)
                    return func_or_class

                return deco
            else:
                # Register an object by function call
                self._do_register(args[0].__name__, args[0])
        elif len(args) == 0:
            # Register an object by decorator
            def deco(func_or_class: object) -> object:
                self._do_register(func_or_class.__name__, func_or_class)
                return func_or_class

            return deco
        else:
            raise ValueError("Do not support the usage!")

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))
        return ret

    def list_keys(self) -> List:
        return list(self._obj_map.keys())

    def __repr__(self) -> str:
        s = "{name}(keys={keys})".format(name=self._name, keys=self.list_keys())
        return s

    def create(self, name: str, *args, **kwargs) -> object:
        """
        Create the class object with the given args and kwargs
        Parameters
        ----------
        name
            The name in the registry
        args
        kwargs
        Returns
        -------
        ret
            The created object
        """
        return self.get(name)(*args, **kwargs)

    def create_with_json(self, name: str, json_str: str):
        """
        Parameters
        ----------
        name
        json_str
        Returns
        -------
        """
        try:
            args = json.loads(json_str)
        except JSONDecodeError:
            raise ValueError('Unable to decode the json string: json_str="{}"'.format(json_str))
        if isinstance(args, (list, tuple)):
            return self.create(name, *args)
        elif isinstance(args, dict):
            return self.create(name, **args)
        else:
            raise NotImplementedError(
                "The format of json string is not supported! We only support "
                'list/dict. json_str="{}".'.format(json_str)
            )
