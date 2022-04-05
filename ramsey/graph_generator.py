"""graph generation"""

import dgl
import numpy as np
import torch


def complete_graph(n_nodes):
    """An auxiliary function define a complete graph"""
    adj = torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes)

    src, dst = adj.nonzero().transpose(0, 1).reshape(2, -1)
    graph = dgl.graph((src, dst))

    return graph


def random_graph(n_nodes, n_edges=None):
    """A random graph generator having a prescribed number of edges."""

    if n_edges is None:
        n_edges = np.random.randint(n_nodes,
                                    np.ceil(n_nodes * (n_nodes + 1) / 2))

    # generate random directed graph
    graph = dgl.rand_graph(n_nodes, n_edges)

    # remove self-loops
    graph = dgl.remove_self_loop(graph)

    # transform it into an undirected simple graph
    graph = dgl.to_bidirected(graph)

    return graph


def complement_graph(graph):
    """Construction of the complement of a given graph."""

    n_nodes = graph.num_nodes()

    # we define the complement graph adjacency matrix from the one of the matrix
    adj = graph.adj().to_dense()

    # define a complete graph adjacency matrix
    complete_graph_adj = torch.ones_like(adj) - torch.eye(n_nodes)

    # the adjacency matrix of the complement graph is the difference of them
    complement_graph_adj = complete_graph_adj - adj

    src, dst = complement_graph_adj.nonzero().transpose(0, 1).reshape(2, -1)
    complement = dgl.graph((src, dst), num_nodes=n_nodes)

    return complement
