import json
import collections
import mxnet as mx
from mxnet import gluon
from autogluon.core.space import Categorical, Space, _strip_config_space

__all__ = ['enas_unit', 'enas_net',
           'Zero_Unit', 'ENAS_Unit', 'ENAS_Sequential']


def enas_unit(**kwvars):
    def registered_class(Cls):
        class enas_unit(ENAS_Unit):
            def __init__(self, *args, **kwargs):
                kwvars.update(kwargs)
                with_zero=False
                if 'with_zero' in kwvars:
                    with_zero = kwvars.pop('with_zero')
                blocks = []
                self._args = []
                for arg in self.get_config_grid(kwvars):
                    blocks.append(Cls(*args, **arg))
                    self._args.append(json.dumps(arg))
                if with_zero:
                    self._args.append(None)
                super().__init__(*blocks, with_zero=with_zero)

            @property
            def node(self):
                arg = self._args[self.index]
                if arg is None: return arg
                summary = {}
                name = self.module_list[self.index].__class__.__name__ + '('
                for k, v in json.loads(arg).items():
                    if 'kernel' in k.lower():
                        cm = ("#8dd3c7", "#fb8072", "#ffffb3", "#bebada", "#80b1d3",
                              "#fdb462", "#b3de69", "#fccde5")
                        summary['fillcolor'] = cm[v]
                    k = k[:1].upper() if len(k) > 4 else k
                    name += '{}{}.'.format(k, v)
                name += ')'
                summary['label'] = name
                return summary

            @staticmethod
            def get_config_grid(dict_space):
                param_grid = {}
                constants = {}
                for k, v in dict_space.items():
                    if isinstance(v, Categorical):
                        param_grid[k] = v.data
                    elif isinstance(v, Space):
                        raise NotImplementedError
                    else:
                        constants[k] = v
                from sklearn.model_selection import ParameterGrid
                configs = list(ParameterGrid(param_grid))
                for config in configs:
                    config.update(constants)
                return configs

        return enas_unit
    return registered_class

def enas_net(**kwvars):
    def registered_class(Cls):
        class ENAS_Net(Cls):
            def __init__(self, *args, **kwargs):
                kwvars.update(kwargs)
                super().__init__(*args, **kwvars)
                #
                self._modules = {}
                self._kwspaces = collections.OrderedDict()
                for k, module in kwvars.items():
                    if isinstance(module, (ENAS_Unit, ENAS_Sequential)):
                        self._modules[k] = module
                        if isinstance(module, ENAS_Unit):
                            self._kwspaces[k] = module.kwspaces
                        else:
                            assert isinstance(module, ENAS_Sequential)
                            for key, v in module.kwspaces.items():
                                new_key = '{}.{}'.format(k, key)
                                self._kwspaces[new_key] = v
                self.latency_evaluated = False
                self._avg_latency = 1

            @property
            def nparams(self):
                nparams = 0
                for k, op in self._modules.items():
                    if isinstance(op, (ENAS_Unit, ENAS_Sequential)):
                        nparams += op.nparams
                    else:
                        # standard block
                        for _, v in op.collect_params().items():
                            nparams += v.data().size
                return nparams

            @property
            def nodeend(self):
                return list(self._modules.keys())[-1]

            @property
            def nodehead(self):
                return list(self._modules.keys())[0]

            @property
            def graph(self):
                from graphviz import Digraph
                e = Digraph(node_attr={'color': 'lightblue2', 'style': 'filled', 'shape': 'box'})
                pre_node = 'input'
                e.node(pre_node)
                for k, op in self._modules.items():
                    if hasattr(op, 'graph'):
                        e.subgraph(op.graph)
                        e.edge(pre_node, op.nodehead)
                        pre_node = op.nodeend
                    else:
                        if hasattr(op, 'node'):
                            if op.node is None: continue
                            node_info = op.node
                        else:
                            node_info = {'label': op.__class__.__name__}
                        e.node(k, **node_info)
                        e.edge(pre_node, k)
                        pre_node = k
                return e


            @property
            def kwspaces(self):
                return self._kwspaces

            def sample(self, **configs):
                striped_keys = [k.split('.')[0] for k in configs.keys()]
                for k in striped_keys:
                    if isinstance(self._modules[k], ENAS_Unit):
                        self._modules[k].sample(configs[k])
                    else:
                        sub_configs = _strip_config_space(configs, prefix=k)
                        self._modules[k].sample(**sub_configs)

            @property
            def latency(self):
                if not self.latency_evaluated:
                    raise Exception('Latency is not evaluated yet.')
                return self._avg_latency

            @property
            def avg_latency(self):
                if not self.latency_evaluated:
                    raise Exception('Latency is not evaluated yet.')
                return self._avg_latency

            def evaluate_latency(self, x):
                # evaluate submodule latency
                for k, op in self._modules.items():
                    if hasattr(op, 'evaluate_latency'):
                        x = op.evaluate_latency(x)
                # calc avg_latency
                avg_latency = 0.0
                for k, op in self._modules.items():
                    if hasattr(op, 'avg_latency'):
                        avg_latency += op.avg_latency
                self._avg_latency = avg_latency
                self.latency_evaluated = True

        return ENAS_Net
    return registered_class

class ENAS_Sequential(gluon.HybridBlock):
    def __init__(self, *modules_list):
        """
        Args:
            modules_list(list of ENAS_Unit)
        """
        super().__init__()
        if len(modules_list) == 1 and isinstance(modules_list, (list, tuple)):
            modules_list = modules_list[0]
        self._modules = {}
        self._blocks = gluon.nn.HybridSequential()
        self._kwspaces = collections.OrderedDict()
        for i, op in enumerate(modules_list):
            self._modules[str(i)] = op
            with self._blocks.name_scope():
                self._blocks.add(op)
            if hasattr(op, 'kwspaces'):
                self._kwspaces[str(i)] = op.kwspaces
        self.latency_evaluated = False
        self._avg_latency = 1

    def __getitem__(self, index):
        return self._blocks[index]

    def hybrid_forward(self, F, x):
        for k, op in self._modules.items():
            x = op(x)
        return x

    def prune(self):
        _modules_keys = self._modules.keys()
        module_list = []
        for k in _modules_keys:
            if isinstance(self._modules[k], ENAS_Unit):
                index = self._modules[k].index
                op = self._modules[k].module_list[index]
                module_list.append(op)
            else:
                module_list.append(self._modules[k])
        return module_list

    @property
    def nodeend(self):
        return list(self._modules.keys())[-1]

    @property
    def nodehead(self):
        return list(self._modules.keys())[0]

    @property
    def graph(self):
        from graphviz import Digraph
        e = Digraph(node_attr={'color': 'lightblue2', 'style': 'filled', 'shape': 'box'})
        pre_node = None
        for i, op in self._modules.items():
            if hasattr(op, 'graph'):
                e.subgraph(op.graph)
                if pre_node:
                    e.edge(pre_node, op.nodehead)
                pre_node = op.nodeend
            else:
                if hasattr(op, 'node'):
                    if op.node is None: continue
                    node_info = op.node
                else:
                    node_info = {'label': op.__class__.__name__}
                e.node(i, **node_info)
                if pre_node:
                    e.edge(pre_node, i)
                pre_node = i
        return e

    @property
    def kwspaces(self):
        return self._kwspaces

    @property
    def nparams(self):
        nparams = 0
        for k, op in self._modules.items():
            if isinstance(op, ENAS_Unit):
                nparams += op.nparams
            else:
                # standard block
                for _, v in op.collect_params().items():
                    nparams += v.data().size
        return nparams

    @property
    def latency(self):
        if not self.latency_evaluated:
            raise Exception('Latency is not evaluated yet.')
        latency = 0.0
        for k, op in self._modules.items():
            if hasattr(op, 'latency'):
                latency += op.latency
        return latency

    @property
    def avg_latency(self):
        if not self.latency_evaluated:
            raise Exception('Latency is not evaluated yet.')
        return self._avg_latency

    def evaluate_latency(self, x):
        # evaluate submodule latency
        for k, op in self._modules.items():
            if hasattr(op, 'evaluate_latency'):
                x = op.evaluate_latency(x)
            else:
                x = op(x)
        # calc avg_latency
        avg_latency = 0.0
        for k, op in self._modules.items():
            if hasattr(op, 'avg_latency'):
                avg_latency += op.avg_latency
        self._avg_latency = avg_latency
        self.latency_evaluated = True
        return x

    def sample(self, **configs):
        for k, v in configs.items():
            self._modules[k].sample(v)

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        for i, op in self._modules.items():
            reprstr += '\n\t{}: {}'.format(i, op)
        reprstr += ')\n'
        return reprstr

    def export(self, path):
        """Export HybridBlock to json format that can be loaded by
        `gluon.SymbolBlock.imports`, `mxnet.mod.Module` or the C++ interface.

        .. note:: When there are only one input, it will have name `data`. When there
                  Are more than one inputs, they will be named as `data0`, `data1`, etc.

        Parameters
        ----------
        path : str
            Path to save model. Two files `path-symbol.json` and `path.params`
            will be created.

        Examples
        --------

        >>> mynet.export('enas')
        >>> mynet_static = mx.gluon.nn.SymbolBlock.imports(
            "enas-symbol.json", ['data'], "enas.params")
        >>> x = mx.nd.random.uniform(shape=(1, 1, 28, 28))
        >>> y = mynet_static(x)
        """
        from mxnet import npx as _mx_npx
        from mxnet.util import is_np_array
        from mxnet import ndarray
        if not self._cached_graph:
            raise RuntimeError(
                "Please first call block.hybridize() and then run forward with "
                "this block at least once before calling export.")
        sym = self._cached_graph[1]
        sym.save('%s-symbol.json'%path, remove_amp_cast=True)

        arg_names = set(sym.list_arguments())
        aux_names = set(sym.list_auxiliary_states())
        arg_dict = {}
        for name, param in self.collect_params().items():
            if name in arg_names:
                arg_dict['arg:%s'%name] = param._reduce()
            elif name in aux_names:
                arg_dict['aux:%s'%name] = param._reduce()
            else:
                pass
        save_fn = _mx_npx.save if is_np_array() else ndarray.save
        save_fn('%s.params'%(path), arg_dict)

class Zero_Unit(gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        return x
    def __repr__(self):
        return self.__class__.__name__

class ENAS_Unit(gluon.HybridBlock):
    def __init__(self, *ops, with_zero=False):
        super().__init__()
        self.module_list = gluon.nn.HybridSequential()
        self._latency = []
        for op in ops:
            self.module_list.add(op)
            self._latency.append(1)
        if with_zero:
            self.module_list.add(Zero_Unit())
            self._latency.append(1)
        self.index = 0
        self._latency_benchmark_times = 10
        self._latency_warmup_times = 5
        self.latency_evaluated = False

    def hybrid_forward(self, F, x):
        return self.module_list[self.index](x)

    @property
    def kwspaces(self):
        return Categorical(*list(range(len(self.module_list))))

    @property
    def nparams(self):
        nparams = 0
        for _, v in self.module_list[self.index].collect_params().items():
            nparams += v.data().size
        return nparams

    @property
    def latency(self):
        if not self.latency_evaluated:
            raise Exception('Latency is not evaluated yet.')
        return self._latency[self.index]

    @property
    def avg_latency(self):
        if not self.latency_evaluated:
            raise Exception('Latency is not evaluated yet.')
        return sum(self._latency) / len(self._latency)

    def evaluate_latency(self, x):
        import time
        for i, op in enumerate(self.module_list):
            latency_i = 0
            for j in range(self._latency_benchmark_times + self._latency_warmup_times):
                start_time = time.time() * 1000 # ms
                #print('op {}, shape x {}'.format(op, x.shape))
                y = op(x)
                mx.nd.waitall()
                end_time = time.time() * 1000 # ms
                if j > self._latency_warmup_times:
                    latency_i += end_time - start_time
            self._latency[i] = latency_i / self._latency_benchmark_times
        self.latency_evaluated = True
        return y

    def sample(self, ind):
        self.index = ind

    def __len__(self):
        return len(self.module_list)

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(num of choices: {}), current architecture:\n\t {}' \
            .format(len(self.module_list), self.module_list[self.index])
        return reprstr
