import torch
from torch import nn, Tensor
from torch_geometric.nn.conv import MessagePassing
import torch_geometric as pyg
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot, zeros

"""
Issues when constructing the RGCN implementation:

Looping over the relation index is not efficient.
"""


class RGCNConv(MessagePassing):
    def __init__(self):
        super().__init__()
        self.self_W = nn.Linear(in_features=self.in_channels, out_features=self.out_channels, bias=False)
        self.W_r = nn.Parameter(torch.randn((self.n_relations, self.in_channels, self.out_channels)))

    def forward(self, x: Tensor, edge_idx, edge_type) -> Tensor:
        self_transform = self.self_W(x)

        # First add the self-transform
        out = self.self_W(x)

        for relation in range(self.n_relations):
            mask = edge_type == relation
            masked_edge_index = edge_idx[:, mask]
            from_neighbors = self.propagate(masked_edge_index, x=x, size=mask.sum())
            out += (from_neighbors @ self.W_r[relation])

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

class RGCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_relations, n_entities):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(n_relations, in_channels, out_channels))
        self.W_0 = nn.Linear(in_channels, out_channels, bias=False)
        self.cs = nn.Parameter(torch.Tensor(n_entities, n_relations))

    def forward(self, x, edge_index):
        # x: (batch, n_entities, in_channels)
        cs_reciprocal = 1 / self.cs
        src = torch.einsum('er,rio,ej -> eo', cs_reciprocal, self.W, x)

        res = scatter(src, )
        x_next = torch.sigmoid(self.W_0(x) + )