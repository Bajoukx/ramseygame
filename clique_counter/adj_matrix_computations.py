"""In this script the cliques that each node is part of is computed using adjacency matrix multiplications"""

import dgl
import numpy as np
import torch

from absl import app
from absl import logging
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_nodes", 10, "Number of nodes that the graph should have.")
flags.DEFINE_integer("n_edges", None, "Number of edges of the graph.")
flags.DEFINE_integer("max_clique_size", None, "Size of cliques to be computed to.")


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


def next_clique_multiplication(cliques, n_nodes=None):
    """Function constructing the k+1 cliques tensor from the k cliques one"""

    if n_nodes is None:
        n_nodes = cliques.shape[0]

    k = len(cliques.shape)
    reshape_entries = (k + 1) * [n_nodes]

    next_size_cliques = cliques.clone()
    cliques = cliques.clone()  # not sure if it is needed
    for i in range(k + 1):
        reshape_entries[i] = 1
        cliques = cliques.reshape(reshape_entries)
        next_size_cliques = next_size_cliques * cliques
        reshape_entries[i] = n_nodes

    return next_size_cliques


def k_cliques(graph, max_clique_size=None):
    """Function computing k_cliques iteratively"""
    n_nodes = graph.num_nodes()
    if max_clique_size is None:
        max_clique_size = np.inf

    # define the 2_cliques as the entries in the adjacency matrix
    cliques = {2: graph.adj().to_dense()}

    # define a stop condition to activate when there is no k_clique in a graph
    there_are_cliques = (cliques[2].any() != 0)
    k = 2

    while there_are_cliques:
        k += 1
        cliques[k] = next_clique_multiplication(cliques[k - 1], n_nodes=n_nodes)
        there_are_cliques = (cliques[k].any() != 0)
        if not there_are_cliques:
            logging.info('The Ramsey number of the graph is %d', k - 1)
        there_are_cliques = there_are_cliques and (k <= max_clique_size)

    return cliques


def cliques_counter(graph, max_clique_size=None):
    """Compute the number of k cliques each node is part of"""
    n_nodes = graph.num_nodes()
    cliques = k_cliques(graph, max_clique_size=max_clique_size)
    nr_cliques_computed = len(cliques.keys())

    if max_clique_size is None:
        max_clique_size = n_nodes

    all_cliques = [n_nodes * [0.] for _ in range(2, max_clique_size + 1)]
    for clique in range(2, nr_cliques_computed + 1):
        for node in range(n_nodes):
            all_cliques[clique - 2][node] = torch.sum(cliques[clique][node]).item() / (
                np.math.factorial(int(clique) - 1))

    return torch.Tensor(all_cliques)


def cliques_graph_and_dual(graph, max_clique_size=None):
    """Function to return the number of cliques of a graph and its complement"""
    dual_graph = complement_graph(graph)
    return cliques_counter(graph, max_clique_size=max_clique_size), \
           cliques_counter(dual_graph, max_clique_size=max_clique_size)


def main(argv):
    g = random_graph(FLAGS.n_nodes, FLAGS.n_edges)
    # g = complete_graph(FLAGS.n_nodes)
    cliques, complement_cliques = cliques_graph_and_dual(g, FLAGS.max_clique_size)
    print(cliques)
    print(complement_cliques)
    # print(k_cliques(g)[2])


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
