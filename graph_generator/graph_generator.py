"""Class for graph generation"""
from scipy.sparse import csr_matrix
import torch as nn
from torch_geometric.data import Data
import torch_geometric


class RandomGraph(Data):
    """Random graph generator"""

    def __init__(self, nr_nodes, **kwargs):
        super().__init__(**kwargs)
        self.nr_nodes = nr_nodes
        self.adj = self.random_edges
        self.x = nn.Tensor([range(self.nr_nodes)]).reshape(self.nr_nodes, 1)
        self.edge_index = torch_geometric.utils.from_scipy_sparse_matrix(
            self.adj)

    @property
    def random_edges(self):
        adj_mat = nn.randint(0, 2, (self.nr_nodes, self.nr_nodes))
        adj_mat_t = nn.triu(adj_mat, diagonal=1)
        adj_mat = adj_mat + nn.transpose(adj_mat_t, 0, 1)
        return csr_matrix(adj_mat)
