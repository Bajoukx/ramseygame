"""Reward functions for the Ramsey game."""

import networkx


def dual_based_reward(graph):
    """Biggest clique number minus the biggest clique in the dual."""
    biggest_clique = networkx.graph_clique_number(graph)
    complement = networkx.complement(graph)
    biggest_clique_in_dual = networkx.graph_clique_number(complement)
    return biggest_clique - biggest_clique_in_dual


def ramsey_number(graph):
    """Maximum clique number in the graph or it's dual."""
    biggest_clique = networkx.graph_clique_number(graph)
    complement = networkx.complement(graph)
    biggest_clique_in_dual = networkx.graph_clique_number(complement)
    return max(biggest_clique, biggest_clique_in_dual)
