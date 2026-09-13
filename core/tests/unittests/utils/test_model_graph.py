import pickle
import random

import pytest

from autogluon.core.utils.model_graph import ModelGraph

nx = pytest.importorskip("networkx")


def _random_dag(seed: int, n_nodes: int = 12, edge_prob: float = 0.3):
    """The same random DAG as a `ModelGraph` and a `networkx.DiGraph`, with node attributes."""
    rng = random.Random(seed)
    names = [f"m{i}" for i in range(n_nodes)]
    rng.shuffle(names)
    graph, reference = ModelGraph(), nx.DiGraph()
    for level, name in enumerate(names):
        attributes = {"level": level, "fit_time": float(level)} if level % 3 else {"level": level}
        graph.add_node(name, **attributes)
        reference.add_node(name, **attributes)
    for i, source in enumerate(names):
        for target in names[i + 1 :]:
            if rng.random() < edge_prob:
                graph.add_edge(source, target)
                reference.add_edge(source, target)
    return graph, reference, names


@pytest.mark.parametrize("seed", range(20))
def test_model_graph_matches_networkx(seed):
    graph, reference, names = _random_dag(seed)

    assert list(graph.nodes) == list(reference.nodes)
    assert set(graph.edges()) == set(reference.edges)
    assert graph.get_node_attributes("fit_time") == nx.get_node_attributes(reference, "fit_time")
    assert graph.lexicographical_topological_sort() == list(nx.lexicographical_topological_sort(reference))
    for name in names:
        assert name in graph.nodes and graph.nodes[name] == reference.nodes[name]
        assert graph.ancestors(name) == nx.ancestors(reference, name)
        assert graph.descendants(name) == nx.descendants(reference, name)
        assert set(graph.ancestors_with_self(name)) == set(nx.bfs_tree(reference, name, reverse=True))
        assert graph.ancestors_with_self(name)[0] == name
        assert set(graph.predecessors(name)) == set(reference.predecessors(name))
        assert set(graph.successors(name)) == set(reference.successors(name))
        assert set(graph.in_edges(name)) == set(reference.in_edges(name))
        for other in names:
            assert graph.has_path(name, other) == nx.has_path(reference, name, other)

    roots = [name for name in names if not list(graph.predecessors(name))]
    # networkx orders a layer by set iteration; the layers' membership is what is defined.
    assert [set(layer) for layer in graph.bfs_layers(roots)] == [
        set(layer) for layer in nx.bfs_layers(reference, roots)
    ]
    assert graph.shortest_path_lengths() == {
        source: dict(lengths) for source, lengths in nx.shortest_path_length(reference)
    }

    keep = names[::2]
    subgraph = graph.subgraph(keep)
    reference_subgraph = nx.subgraph(reference, keep)
    assert set(subgraph.nodes) == set(reference_subgraph.nodes)
    assert set(subgraph.edges()) == set(reference_subgraph.edges)
    assert subgraph.lexicographical_topological_sort() == list(nx.lexicographical_topological_sort(reference_subgraph))


def test_model_graph_mutation_and_copy():
    graph, reference, names = _random_dag(seed=3)
    copied = graph.copy()
    copied.nodes[names[0]]["level"] = -1
    assert graph.nodes[names[0]]["level"] != -1, "a copy has its own attribute dicts"

    victim = names[len(names) // 2]
    graph.remove_node(victim)
    reference.remove_node(victim)
    assert set(graph.nodes) == set(reference.nodes) and set(graph.edges()) == set(reference.edges)

    survivor = names[-1]
    graph.remove_edges_from(list(graph.in_edges(survivor)))
    reference.remove_edges_from(list(reference.in_edges(survivor)))
    assert set(graph.edges()) == set(reference.edges)
    assert list(graph.predecessors(survivor)) == []

    graph.add_node(survivor, level=99)
    assert graph.nodes[survivor]["level"] == 99, "adding an existing node updates its attributes"

    restored = pickle.loads(pickle.dumps(graph))
    assert set(restored.edges()) == set(graph.edges()) and restored.nodes[survivor] == graph.nodes[survivor]


def test_model_graph_cycle_is_rejected():
    graph = ModelGraph()
    graph.add_edge("a", "b")
    graph.add_edge("b", "a")
    with pytest.raises(ValueError, match="cycle"):
        graph.lexicographical_topological_sort()
