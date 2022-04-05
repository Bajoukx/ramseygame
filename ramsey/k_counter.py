"""Script computing k cliques iteratively.

This version was created because computing tensors with the k_cliques is too
heavy without using sparse tensors. It computes the k cliques by
induction. Cliques of size 2 are given by the edges of a graph, and for every k,
the k + 1 cliques are computed as:

for each node:
  for each neighbour of the node:
    k clique = {node} union {k - 1 other nodes}
    and
    k clique = {neighbour} union {k - 1 other nodes}
    then
    k + 1 clique = {node, neighbour} union {k - 1 other nodes}

"""

import dgl
import torch
from scipy.special import comb
from absl import app
from absl import logging
from absl import flags

from graph_generator import random_graph, complement_graph  # complete_graph

FLAGS = flags.FLAGS
flags.DEFINE_integer('n_nodes', 10,
                     'Number of nodes that the graph should have.')
flags.DEFINE_integer('n_edges', None, 'Number of edges of the graph.')
flags.DEFINE_integer('max_clique_size', None,
                     'Size of cliques to be computed to.')


def list_of_neighbours(graph, node=None):
    """Function returning a list of neighbours of each node."""

    assert isinstance(graph, dgl.DGLGraph), 'expected a dgl graph'

    if node is None:
        lst = []
        for i in range(graph.num_nodes()):
            node_list = graph.successors(i)
            lst.append(sorted([[x.item()] for x in node_list]))
        return lst
    else:
        assert isinstance(node, int), f'expected {node} to be an integer'
        assert 0 <= node < graph.num_nodes(),\
            f'expected {node} to be between 0 and {graph.num_nodes()}'
        node_list = graph.successors(node)
        return sorted([x.item() for x in node_list])


def k_to_k_plus_one(graph, k_list):
    """This is the induction step"""
    list_k_plus_one_cliques = []
    for node_idx in range(graph.num_nodes()):
        k_plus_one_cliques_in_node = []
        node_neighbours = list_of_neighbours(graph, node_idx)

        for neighbour in node_neighbours:
            k_tuples = [tuple(x) for x in k_list[neighbour]]
            k_tuples_node = [tuple(x) for x in k_list[node_idx]]
            common_cliques = set(k_tuples).intersection(k_tuples_node)

            for k_clique in common_cliques:
                k_plus_one_clique = sorted(list(k_clique) + [neighbour])

                if k_plus_one_clique not in k_plus_one_cliques_in_node:
                    k_plus_one_cliques_in_node.append(k_plus_one_clique)

        list_k_plus_one_cliques.append(k_plus_one_cliques_in_node)

    return list_k_plus_one_cliques


def cliques_in_graph(graph):
    """Returns a dictionary with the cliques that each node is part of.

    cliques_in_graph[k][i] is a list with the k-1 nodes such that the node i
    and these k-1 nodes form a k clique."""

    cliques = {2: list_of_neighbours(graph)}
    n_nodes = graph.num_nodes()

    for k in range(3, n_nodes + 1):
        cliques[k] = k_to_k_plus_one(graph, cliques[k - 1])

    return cliques


def cliques_counter(graph):
    """The number of k cliques that each node is part of.

    cliques_counter[i, k] is the number of k cliques that the node i is part
    of."""
    cliques = cliques_in_graph(graph)
    n_nodes = graph.num_nodes()
    cliques_count = torch.zeros(n_nodes, n_nodes - 1)
    for i in range(n_nodes):
        for j in range(2, n_nodes + 1):
            cliques_count[i, j - 2] = len(cliques[j][i])
    return cliques_count


def cliques_graph_and_complement(graph):
    """cliques of graph and complement as a (n_nodes) x 2 x (nodes-1) tensor"""
    complement = complement_graph(graph)
    cliques_graph = cliques_counter(graph)
    cliques_complement = cliques_counter(complement)
    cliques_graph = cliques_graph[:, None, :]
    cliques_complement = cliques_complement[:, None, :]
    total_cliques = torch.cat((cliques_graph, cliques_complement), 1)

    return total_cliques


def cliques_as_feature(graph, normalized=True):
    """returns cliques as a graph feature"""

    cliques = cliques_graph_and_complement(graph)
    if normalized:
        # normalized features have values between 0 and 1.
        n_nodes = graph.num_nodes()
        comb_tensor = torch.zeros(2, n_nodes - 1)
        for i in range(n_nodes - 1):
            comb_tensor[:, i] = 1 / comb(n_nodes - 1, i + 1)
        cliques = comb_tensor * cliques
        graph.ndata['cliques'] = cliques
    else:
        graph.ndata['cliques'] = cliques
    return graph


def main(_):
    g = random_graph(FLAGS.n_nodes, FLAGS.n_edges)
    # g = complete_graph(FLAGS.n_nodes)
    print(cliques_as_feature(g, normalized=False).ndata['cliques'])


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
