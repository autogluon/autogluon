import collections

import mxnet as mx
from mxnet import gluon
from ...core.space import *
from ...core.space import Space, _strip_config_space

from ..enas import ENAS_Sequential

__all__ = ['Wire_Stage', 'Wire_Sequential']

class Wire_Stage(gluon.HybridBlock):
    """The Random Wire Stage, each op should work on the same featuremap size.
    """
    def __init__(self, *modules_list, label_graph=True, init_method='sequential'):
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
        # initialize connections
        self._connections = {}
        keys = list(self._modules.keys())
        N = len(self._modules)
        for i in range(N-2):
            for j in range(i+1, N-1):
                if init_method == 'dense':
                    self._connections['{}.{}'.format(keys[i], keys[j])] = 1
                else:
                    assert init_method == 'sequential'
                    self._connections['{}.{}'.format(keys[i], keys[j])] = 1 if j == (i + 1) else 0
                self._kwspaces['{}.{}'.format(keys[i], keys[j])] = Categorical(0, 1)
        assert len(self._connections) == (N - 1) * (N - 2) // 2
        self._active_nodes = list(keys)
        self._invalide_nodes = []
        self._leaf_nodes = [keys[-2]]
        self.keys = keys
        self.label_graph = label_graph

    def sample(self, **configs):
        for k, v in configs.items():
            self._connections[k] = v
        in_node = self.keys[0]
        self._active_nodes = []
        self._leaf_nodes = []
        def mark_connected(cur_node, connections, active_nodes):
            if cur_node in active_nodes: return
            active_nodes.append(cur_node)
            is_leaf = True
            for k, v in connections.items():
                if k.startswith(cur_node) and v:
                    is_leaf = False
                    new_node = k.split('.')[-1]
                    mark_connected(new_node, connections, active_nodes)
            if is_leaf:
                self._leaf_nodes.append(cur_node)
        mark_connected(in_node, self._connections, self._active_nodes)
        self._active_nodes.sort(key=int)# = sorted(self._active_nodes)
        self._leaf_nodes.sort(key=int)# = sorted(self._leaf_nodes)
        self._invalide_nodes = list(set(self.keys) - set(self._active_nodes))
        #return True

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
                        assert in_node in results
                        inputs = inputs + results[in_node] if inputs is not None else results[in_node]
            assert inputs is not None
            results[node] = self._modules[node](inputs)
        inputs = results[self._leaf_nodes[0]]
        for leaf_node in self._leaf_nodes[1:]:
            inputs = inputs + results[leaf_node]
        return self._modules[self.keys[-1]](inputs)

    @property
    def kwspaces(self):
        return self._kwspaces

    @property
    def nodehead(self):
        node_name = self._prefix + '.' + self.keys[0]
        return node_name

    @property
    def nodeend(self):
        node_name = self._prefix + '.' + self.keys[-1]
        return node_name

    @property
    def graph(self):
        from graphviz import Digraph
        e = Digraph(node_attr={'color': 'lightblue2', 'style': 'filled'})#, 'shape': 'box'
        e.attr(size='8,3')
        out_node = self.keys[-1]
        out_node_name = self._prefix + '.' + out_node
        node_label = self._modules[out_node].__class__.__name__ + out_node if self.label_graph else ''
        e.node(out_node_name, label=node_label)
        for node in self._active_nodes:
            node_name = self._prefix + '.' + node
            node_label = self._modules[node].__class__.__name__ + node if self.label_graph else ''
            e.node(node_name, label=node_label)
        for k, v in self._connections.items():
            k1, k2 = k.split('.')
            if v and k1 in self._active_nodes and k2 in self._active_nodes:
                k1_name = self._prefix + '.' + k1
                k2_name = self._prefix + '.' + k2
                e.edge(k1_name, k2_name)
        for k in self._leaf_nodes:
            k_name = self._prefix + '.' + k
            e.edge(k_name, out_node_name)
        return e

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        for key in self._active_nodes:
            op = self._modules[key]
            reprstr += '\n\t{}: {}'.format(key, op)
        key = self.keys[-1]
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
        for k, op in self._modules.items():
            min_config = _strip_config_space(configs, prefix=k)
            op.sample(**min_config)

    @property
    def graph(self):
        from graphviz import Digraph
        e = Digraph(node_attr={'color': 'lightblue2', 'style': 'filled'}) #, 'shape': 'box'
        pre_node = 'input'
        e.node(pre_node)
        for i, op in self._modules.items():
            if hasattr(op, 'graph'):
                e.subgraph(op.graph)
                e.edge(pre_node, op.nodehead)
                pre_node = op.nodeend
            else:
                if hasattr(op, 'node'):
                    if op.node is None: continue
                    node_info = op.node
                else:
                    node_info = {}
                    node_info['label'] = op.__class__.__name__
                e.node(i, **node_info)
                e.edge(pre_node, i)
                pre_node = i
        return e
