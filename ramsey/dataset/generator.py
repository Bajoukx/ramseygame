"""The dataset generator."""

import os

import networkx
from torch.utils.data import Dataset


def compose_file_name(self):
        """Composes the entire file name and path for the data."""
        path = os.path.join(self.file_path, 'ramsey_', self.n_samples)
        for node in self.n_nodes:
            path = os.path.join(path, '_', node)
        path = os.path.join(path, '_', self.algorithm)
        path = os.path.join(path, '.npy')

class CustomGraphDataset(Dataset):

    def __init__(self, n_nodes, edge_probability, n_samples=1000000):
        self.n_nodes = n_nodes
        self.edge_probability = edge_probability
        self.n_samples = n_samples
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        random_graph = networkx.erdos_renyi_graph(self.n_nodes, self.
        edge_probability)
        cliques = list(networkx.enumerate_all_cliques(random_graph))
        clique_number = networkx.graph_clique_number(random_graph, cliques)

        dual = networkx.complement(random_graph)
        dual_cliques = list(networkx.enumerate_all_cliques(dual))
        dual_clique_number = networkx.graph_clique_number(dual, dual_cliques)


        return {
            'graph': list(random_graph),
            'graph_edges': list(random_graph.edges),
            'cliques': cliques,
            'n_edges': random_graph.number_of_edges(),
            'clique_number': clique_number,
            'dual': list(dual),
            'dual_edges': list(dual.edges),
            'dual_clique_number': dual_clique_number,
            'dual_cliques': dual_cliques,
            'dual_clique_number': dual_clique_number,
        }
