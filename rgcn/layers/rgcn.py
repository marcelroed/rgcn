from typing import Literal

import torch
from torch import nn, Tensor
from torch_geometric.nn.conv import MessagePassing
import torch_geometric as pyg
from torch_geometric.typing import OptTensor
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot, zeros
from tqdm import trange
import torch.nn.functional as F

"""
Issues when constructing the RGCN implementation:

Looping over the relation index is not efficient.
"""



class RelLinear(nn.Module):
    def __init__(self, in_channels, out_channels, n_relations):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_relations = n_relations
        self.weights = nn.Parameter(torch.zeros((self.n_relations, self.in_channels, self.out_channels)))
        self.initialize_weights()

    def __getitem__(self, relation_index: int):
        return self.weights[relation_index]

    def initialize_weights(self):
        glorot(self.weights)

    def l2_norm(self):
        return torch.norm(self.weights, dim=(1, 2)).sum()


class DecomposedRelLinear(nn.Module):
    def __init__(self, in_features, out_features, num_relations, num_bases):
        super().__init__()
        self.bases = nn.Parameter(torch.zeros(num_bases, in_features, out_features))
        self.base_weights = nn.Parameter(torch.randn(num_relations, num_bases))
        self.initialize_weights()

    def initialize_weights(self):
        glorot(self.bases)
        self.base_weights.data = torch.ones_like(self.base_weights) / self.base_weights.shape[1]
        # F.normalize(self.base_weights, dim=1, out=self.base_weights)

    def __getitem__(self, relation_index: int):
        return torch.einsum('b,bio->io', self.base_weights[relation_index], self.bases)

    def l2_norm(self):
        return torch.norm(self.bases, dim=(1, 2)).sum()


class RGCNDirectConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_relations, decomposition_method: Literal['none', 'basis', 'block'], n_basis_functions):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_relations = n_relations
        self.n_basis_functions = n_basis_functions
        self.self_W = nn.Linear(in_features=self.in_channels, out_features=self.out_channels, bias=False)
        if decomposition_method == 'none':
            self.W_r = RelLinear(in_channels=self.in_channels, out_channels=self.out_channels,
                                 n_relations=self.n_relations)
        elif decomposition_method == 'basis':
            self.W_r = DecomposedRelLinear(in_features=self.in_channels, out_features=self.out_channels,
                                           num_relations=self.n_relations, num_bases=self.n_basis_functions)

    def forward(self, x: OptTensor, edge_idx, edge_type, normalization_constants):
        if x is None:
            # inputted x is the identity matrix
            self_transform = self.self_W.weight.T
        else:
            self_transform = self.self_W(x)

        out = torch.zeros_like(self_transform)  # self_transform.clone()

        for relation in range(self.n_relations):
            relation_edges = edge_idx[:, edge_type == relation]
            if x is None:
                relation_transform = self.W_r.__getitem__(relation)
            else:
                relation_transform = F.linear(x, self.W_r.__getitem__(relation).T, None)

            scatter_from = relation_transform[relation_edges[0]]
            if normalization_constants is not None:
                scatter_from /= normalization_constants[relation, relation_edges[1]].view(-1, 1).float()
            scatter(scatter_from, relation_edges[1], dim=0, out=out)

        out += self_transform

        return out

    def l2_norm(self):
        return self.self_W.weight.norm(p=2) + self.W_r.l2_norm()



class RGCNConv(MessagePassing):
    propagate_type = {'x': torch.Tensor}

    def __init__(self, in_channels, out_channels, n_relations):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_relations = n_relations
        self.self_W = nn.Linear(in_features=self.in_channels, out_features=self.out_channels, bias=False)
        self.W_r = nn.Parameter(torch.randn((self.n_relations, self.in_channels, self.out_channels)))

    def forward(self, x, edge_idx, edge_type) -> Tensor:
        # First add the self-transform
        if x is None:
            out = self.self_W.weight
        else:
            out = self.self_W(x)

        for relation in range(self.n_relations):
            mask = edge_type == relation
            masked_edge_index = edge_idx[:, mask]
            from_neighbors = self.propagate(edge_index=masked_edge_index, x=x, size=(x.size(0), x.size(0)))
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

        # res = scatter(src, )
        # x_next = torch.sigmoid(self.W_0(x) + )
