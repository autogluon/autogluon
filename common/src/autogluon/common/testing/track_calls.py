import functools

import pandas as pd


# TODO: Adjust code doc
# TODO: Use this for AbstractModel to simplify integration of new models so users can check correctness!
# TODO: Ex with qualname: "Missing call to AbstractModel.fit, maybe it is overwritten?"
def track_all_calls(cls):
    """ Recursively applies call tracking to a class and its parent classes """

    original_methods = {}
    # Store original methods to avoid double wrapping if parent classes are also processed separately
    for key, value in vars(cls).items():
        if callable(value):
            original_methods[key] = value

    # Function to wrap methods
    def count_calls(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "_method_calls"):
                self._method_calls = {}
                self._method_call_order = []
                self._total_calls = 0
                self._method_depth = 0
            # name = method.__name__
            name = method.__qualname__
            if name not in self._method_calls:
                self._method_calls[name] = 0
            self._total_calls += 1
            cur_depth = self._method_depth
            self._method_depth += 1
            cur_calls = self._total_calls
            self._method_calls[name] += 1
            pos = len(self._method_call_order)
            parent_name = None
            if cur_depth != 0:
                for i in range(1, pos+1):
                    i = -i
                    if self._method_call_order[i][3] == (cur_depth - 1):
                        parent_name = self._method_call_order[i][0]
                        # parent_cls = parent_name.split(".", 1)[0]
                        break
            else:
                parent_name = "root"
            if parent_name is None:
                raise AssertionError(f"Parent name cannot be None.")
            cur_cls, cur_method = name.split(".", 1)

            self._method_call_order.append([
                name,
                parent_name,
                self._method_calls[name],
                cur_depth,
                cur_cls,
                cur_method,
            ])

            out = method(self, *args, **kwargs)
            self._method_depth -= 1
            num_calls_between = self._total_calls - cur_calls


            self._method_call_order[pos] += [num_calls_between, self._total_calls - 1]
            return out
        return wrapper

    # Wrap each method of the class
    for key, value in original_methods.items():
        setattr(cls, key, count_calls(value))

    # Apply recursively to parent classes
    for base in cls.__bases__:
        if base is not object:  # Avoid modifying the base 'object' class
            track_all_calls(base)

    if not hasattr(cls, 'get_call_counts'):
        # Method to get call counts
        def get_call_counts(self):
            return self._method_calls

        def get_call_order(self):
            col_names = ["class_method", "parent_method", "num_calls", "depth", "class", "method", "child_calls", "exit_idx"]
            return pd.DataFrame(self._method_call_order, columns=col_names)

        setattr(cls, 'get_call_counts', get_call_counts)
        setattr(cls, 'get_call_order', get_call_order)

    return cls


def check_status(call_order: pd.DataFrame, model):
    pass

    # custom_mem_estimate =
