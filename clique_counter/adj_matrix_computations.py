"""In this script the cliques that each node is part of is computed using adjacency matrix multiplications"""

import dgl
import numpy as np

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_nodes", 10, "Number of nodes that the graph should have.")
flags.DEFINE_integer("num_edges", None,
                     "Number of nodes.")


def random_graph(n_nodes, n_edges=None):

    if n_edges is None:
        n_edges = np.random.randint(n_nodes, np.ceil(n_nodes*(n_nodes+1)/2))

    graph = dgl.rand_graph(n_nodes, n_edges)

    # remove self-loops
    graph = dgl.remove_self_loop(graph)

    # transform it into an undirected simple graph
    graph = dgl.to_bidirected(graph)

    return graph


# def cliques_3(graph):
#
#     n_nodes = graph.num_nodes()
#     adj = graph.adj().to_dense()
#
#     clique_3 = adj * adj.reshape((n_nodes, n_nodes, 1)) * adj.reshape((n_nodes, 1, n_nodes))
#
#     return clique_3


def cliques_k(graph):
    """function computing k_cliques iteratively"""
    n_nodes = graph.num_nodes()

    # define the 2_cliques as the entries in the adjacency matrix
    cliques = {'2': graph.adj().to_dense()}

    # define a stop condition to be when there is no k_clique in a graph
    there_are_cliques = (cliques['2'].any() != 0)
    k = 2
    while there_are_cliques:
        k += 1
        reshape_entries = k * [n_nodes]
        new_clique = cliques[str(k - 1)].clone()
        old_clique = cliques[str(k - 1)].clone()
        for i in range(k):
            reshape_entries[i] = 1
            old_clique = old_clique.reshape(reshape_entries)
            new_clique = new_clique * old_clique
            reshape_entries[i] = n_nodes
        cliques[str(k)] = new_clique
        there_are_cliques = (cliques[str(k)].any() != 0)

    return cliques, k-1


def main(argv):
    g = random_graph(FLAGS.n_nodes)
    print(cliques_k(g))


if __name__ == '__main__':
    app.run(main)
