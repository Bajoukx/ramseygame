"""k-cliques auto-encoder."""


import torch.nn as nn

from k_counter import cliques_graph_and_complement
from graph_generator import complete_graph


class Encoder(nn.Module):
    """The encoder goes from k-cliques to a graph.

    Is starts with a complete graph with n nodes where each node has the list of
    k-cliques as features and predicts if each edge should or not belong to
    the graph."""
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, cliques):
        n_nodes = cliques.shape[0]
        graph = complete_graph(n_nodes)
        graph.ndata['cliques'] = cliques
        return


class Decoder(nn.Module):
    """The decoder computes the k_cliques of a graph"""
    def forward(self, graph):
        return cliques_graph_and_complement(graph)


class AutoEncoder(nn.Module):
    """Autoencoder of cliques list of agraph and its complement graph"""
    def __init__(self, cliques):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, cliques):
        graph = self.encoder(cliques)
        cliques = self.encoder(graph)
        return cliques
