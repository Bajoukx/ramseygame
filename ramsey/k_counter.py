"""This is not a message passing layer yet"""

import torch
from absl import app
from absl import logging
from absl import flags

from graph_generator import random_graph, complement_graph

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_nodes", 10, "Number of nodes that the graph should have.")
flags.DEFINE_integer("n_edges", None, "Number of edges of the graph.")
flags.DEFINE_integer("max_clique_size", None, "Size of cliques to be computed to.")


def list_of_neighbours(graph, node=None):
    """function returning a list of neighbours of each node"""
    if node is None:
        lst = []
        for i in range(graph.num_nodes()):
            node_list = graph.successors(i)
            lst.append(sorted([[x.item()] for x in node_list]))
        return lst
    else:
        node_list = graph.successors(node)
        return sorted([x.item() for x in node_list])


def k_to_k_plus_one(graph, k_list):
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
    cliques = {2: list_of_neighbours(graph)}
    n_nodes = graph.num_nodes()
    for k in range(3, n_nodes+1):
        cliques[k] = k_to_k_plus_one(graph, cliques[k-1])
    return cliques


def cliques_counter(graph):
    cliques = cliques_in_graph(graph)
    n_nodes = graph.num_nodes()
    cliques_count = torch.zeros(n_nodes, n_nodes-1)
    for i in range(n_nodes):
        for j in range(2, n_nodes+1):
            cliques_count[i, j-2] = len(cliques[j][i])
    return cliques_count


def main(argv):
    g = random_graph(FLAGS.n_nodes, FLAGS.n_edges)
    # print(cliques_in_graph(g))
    print(cliques_counter(g))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)

