import collections

import mxnet as mx
from mxnet import gluon
from ...core.space import *
from ...core.space import Space, _strip_config_space

from ..enas import ENAS_Sequential

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = ['Wire_Stage']

class Wire_Stage(gluon.HybridBlock):
    """The Random Wire Stage, each op should work on the same featuremap size.
    """
    def __init__(self, *modules_list, init_method='sequential'):
        """
        Args:
            modules_list(list of Block)
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
        self.latency_evaluated = False
        self._avg_latency = 1
        # initialized as all connected
        self._connections = {}
        keys = list(self._modules.keys())
        for i in range(len(self._modules)):
            for j in range(i+1, len(self._modules)):
                if init_method == 'dense':
                    self._connections['{}.{}'.format(keys[i], keys[j])] = 1
                else:
                    assert init_method == 'sequential'
                    self._connections['{}.{}'.format(keys[i], keys[j])] = 1 if j == (i + 1) else 0
                self._kwspaces['{}.{}'.format(keys[i], keys[j])] = Categorical(0, 1)
        self._active_nodes = list(keys)
        self.reverse_keys = keys
        self.reverse_keys.reverse()

    def sample(self, **configs):
        for k, v in configs.items():
            self._connections[k] = v
        # calc active nodes
        self._active_nodes = []

        def connect2root(node, root, connections, graph_nodes):
            if node in graph_nodes:
                return True
            if node == root:
                graph_nodes.append(node)
                return True
            connected = False
            for k, v in connections.items():
                if k.endswith(node) and v:
                    new_node = k.split('.')[0]
                    if new_node in graph_nodes:
                        connected = True
                    elif connect2root(new_node, root, connections, graph_nodes):
                        connected = True
            if connected:
                graph_nodes.append(node)
            return connected

        return connect2root(self.reverse_keys[0], self.reverse_keys[-1], self._connections,
                            self._active_nodes)

    def hybrid_forward(self, F, x):
        # first node
        node = self._active_nodes[0]
        results = {}
        results[node] = self._modules[node](x)
        for node in self._active_nodes[1:]:
            inputs = None
            for k, v in self._connections.items():
                if k.split('.')[1] == node and v:
                    in_node = k.split('.')[0]
                    if in_node in self._active_nodes:
                        print('k, in_node:', k, in_node)
                        assert in_node in results
                        inputs = inputs + results[in_node] if inputs is not None else results[in_node]
            assert inputs is not None
            results[node] = self._modules[node](inputs)
        return results[self.reverse_keys[0]]

    @property
    def kwspaces(self):
        return self._kwspaces

    @property
    def nodehead(self):
        node_name = self._prefix + '.' + self._active_nodes[0]
        return node_name

    @property
    def nodeend(self):
        node_name = self._prefix + '.' + self._active_nodes[-1]
        return node_name

    @property
    def graph(self):
        from graphviz import Graph
        e = Graph(node_attr={'color': 'lightblue2', 'style': 'filled', 'shape': 'box'})
        e.attr(size='8,3')
        for node in self._active_nodes:
            node_name = self._prefix + '.' + node
            e.node(node_name, label=self._modules[node].__class__.__name__+node)
        for k, v in self._connections.items():
            k1, k2 = k.split('.')
            if v and k1 in self._active_nodes and k2 in self._active_nodes:
                k1_name = self._prefix + '.' + k1
                k2_name = self._prefix + '.' + k2
                e.edge(k1_name, k2_name)
        return e

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        for key in self._active_nodes:
            op = self._modules[key]
            reprstr += '\n\t{}: {}'.format(key, op)
        reprstr += ')\n'
        return reprstr

class Wire_Sequential(ENAS_Sequential):
    def __init__(self, *modules_list):
        """
        Args:
            modules_list(list of ENAS_Unit)
        """
        super().__init__()
        if len(modules_list) == 1 and isinstance(modules_list, (list, tuple)):
            modules_list = modules_list[0]
        self._modules = {}
        self._blocks = mx.gluon.nn.HybridSequential()
        self._kwspaces = collections.OrderedDict()
        for i, op in enumerate(modules_list):
            self._modules[str(i)] = op
            with self._blocks.name_scope():
                self._blocks.add(op)
            if hasattr(op, 'kwspaces'):
                if isinstance(op.kwspaces, Space):
                    self._kwspaces[str(i)] = op.kwspaces
                else:
                    for sub_k, sub_v in op.kwspaces.items():
                        new_k = '{}.{}'.format(str(i), sub_k)
                        self._kwspaces[new_k] = sub_v
        self.latency_evaluated = False
        self._avg_latency = 1

    @property
    def kwspaces(self):
        return self._kwspaces

    def sample(self, **configs):
        rets = []
        for k, op in self._modules.items():
            min_config = _strip_config_space(configs, prefix=k)
            rets.append(op.sample(**min_config))
        return all(rets)
