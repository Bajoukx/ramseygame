"""graph generation"""

import dgl
import numpy as np
import torch


def complete_graph(n_nodes):
    """An auxiliary function define a complete graph"""
    adj = torch.ones(n_nodes, n_nodes)

    src, dst = adj.nonzero().transpose(0, 1).reshape(2, -1)
    graph = dgl.graph((src, dst))
    graph = dgl.remove_self_loop(graph)

    return graph


def random_graph(n_nodes, n_edges=None):
    """A random graph generator having a prescribed number of edges"""

    if n_edges is None:
        n_edges = np.random.randint(n_nodes, np.ceil(n_nodes * (n_nodes + 1) / 2))

    # generate random directed graph
    graph = dgl.rand_graph(n_nodes, n_edges)

    # remove self-loops
    graph = dgl.remove_self_loop(graph)

    # transform it into an undirected simple graph
    graph = dgl.to_bidirected(graph)

    return graph


def complement_graph(graph):
    """A function proving the complement of a given graph"""
    adj = graph.adj().to_dense()
    complete_graph_adj = torch.ones_like(adj)
    complement_graph_adj = complete_graph_adj - adj
    src, dst = complement_graph_adj.nonzero().transpose(0, 1).reshape(2, -1)
    complement = dgl.graph((src, dst))
    complement = dgl.remove_self_loop(complement)
    return complement
