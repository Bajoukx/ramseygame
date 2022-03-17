"""In this script the cliques that each node is part of is computed using adjacency matrix multiplications"""

import dgl
import numpy as np
import torch

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_nodes", 10, "Number of nodes that the graph should have.")
flags.DEFINE_integer("num_edges", None,
                     "Number of nodes.")


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
        n_edges = np.random.randint(n_nodes, np.ceil(n_nodes*(n_nodes+1)/2))

    # generate random directed graph
    graph = dgl.rand_graph(n_nodes, n_edges)

    # remove self-loops
    graph = dgl.remove_self_loop(graph)

    # transform it into an undirected simple graph
    graph = dgl.to_bidirected(graph)

    return graph


def complementary_graph(graph):
    adj = graph.adj().to_dense()
    complete_graph_adj = torch.ones_like(adj)
    complement_graph_adj = complete_graph_adj - adj
    src, dst = complement_graph_adj.nonzero().transpose(0,1).reshape(2, -1)
    complement_graph = dgl.graph((src, dst))
    complement_graph = dgl.remove_self_loop(complement_graph)
    return complement_graph


def k_cliques(graph):
    """function computing k_cliques iteratively"""
    n_nodes = graph.num_nodes()

    # define the 2_cliques as the entries in the adjacency matrix
    cliques = {2: graph.adj().to_dense()}

    # define a stop condition to activate when there is no k_clique in a graph
    there_are_cliques = (cliques[2].any() != 0)
    k = 2
    while there_are_cliques:
        k += 1
        reshape_entries = k * [n_nodes]
        new_clique = cliques[k - 1].clone()
        old_clique = cliques[k - 1].clone()
        for i in range(k):
            reshape_entries[i] = 1
            old_clique = old_clique.reshape(reshape_entries)
            new_clique = new_clique * old_clique
            reshape_entries[i] = n_nodes
        cliques[k] = new_clique
        there_are_cliques = (cliques[k].any() != 0)

    # define the Ramsey number of the graph
    ramsey_number = k-1
    return cliques, ramsey_number


def cliques_counter(graph):
    """Compute the number of k cliques each node is part of"""
    n_nodes = graph.num_nodes()
    cliques, ramsey_number = k_cliques(graph)

    all_cliques = {i: n_nodes*[0.] for i in range(2, n_nodes + 1)}
    for clique in range(2, ramsey_number + 1):
        for i in range(n_nodes):
            all_cliques[clique][i] = torch.sum(cliques[clique][i]).item()/(np.math.factorial(int(clique)-1))

    return all_cliques


def cliques_graph_and_dual(graph):
    """Function to return the number of cliques of a graph and its complement"""
    dual_graph = complementary_graph(graph)
    return cliques_counter(graph), cliques_counter(dual_graph)


def main(argv):
    g = random_graph(FLAGS.n_nodes)
    # g = complete_graph(FLAGS.n_nodes)
    print(cliques_graph_and_dual(g))


if __name__ == '__main__':
    app.run(main)
