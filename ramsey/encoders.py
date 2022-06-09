"""Encoder functions for the ramsey environment.

TODO(@bajoukx): Consider using openai's gym wrappers for this instead. More
details can be found at:
https://alexandervandekleut.github.io/gym-wrappers/
"""

import itertools

import networkx


def graph_hot_encoder_dict(n_nodes):
    """A dictionary that encodes integers into graph edges.

    A undirected graph with n nodes has a maximum of n(n-1)/2 edges. So we can
    encode the integers from 1 to n(n-1)/2 has the i-th edge of the graph. This
    is done in the following way:

       int   |   edge
    ---------------------
    0        | (0, 0)
    1        | (0, 1)
    ...
    n-1      | (0,n-1)
    n        | (1, 1)
    ...
    n(n-1)/2 | (n-1, n-1)

    Generating a complete undirected graph with n nodes gives us all the
    necessary edges. The i-th tuple is then accessed by it's index:
    encoder_dictionary[i]
    """
    return list(networkx.complete_graph(n_nodes).edges)


def one_hot_encode(dictionary, graph_edges):
    """One hot encodes a graph into a binary list."""
    return [element in list(graph_edges) for element in dictionary]


def one_hot_decode(dictionary, binary_list):
    """One hot decodes a binary list into a list of edges."""
    edges = itertools.compress(dictionary, binary_list)
    return edges
