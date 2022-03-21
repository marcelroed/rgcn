import torch
from torch import nn
import torch.nn.functional as F
from dbgpy import dbg


class DistMultDecoder(nn.Module):
    def __init__(self, n_relations, n_channels):
        super().__init__()
        # (n_relations, n_entities) expands to (n_relations, n_entities, n_entities)
        self.R_diagonal = nn.Parameter(torch.randn(n_relations, n_channels))

    def forward(self, x, edge_index, edge_type):
        # x: (batch, n_entities, n_channels)
        # Produce the score f(s, r, o)
        s = F.normalize(x[edge_index[0, :]], dim=1)
        # print(s.shape)
        r = self.R_diagonal[edge_type]
        o = F.normalize(x[edge_index[1, :]], dim=1)
        score = torch.sum(s * r * o, dim=1)
        return score


class ComplExDecoder(nn.Module):
    def __init__(self, n_relations, n_entities):
        super().__init__()
        # (n_relations, n_entities) expands to (n_relations, n_entities, n_entities)
        self.R_diagonal = nn.Parameter(torch.randn(n_relations, n_entities, dtype=torch.complex64))

    def forward(self, x, edge_index, edge_type):
        assert x.dtype == torch.complex64, 'ComplEx doesn\'t make much sense unless the features are complex'
        # x: (batch, n_entities, n_channels)
        # Produce the score f(s, r, o)
        s = x[edge_index[0, :]]
        r = self.R_diagonal[edge_type]
        o = x[edge_index[1, :]]
        score = torch.sum(s * r * o, dim=1)

        return score.real


class RESCAL(nn.Module):
    def __init__(self, n_relations, n_entities):
        super().__init__()
        # (n_relations, n_entities) expands to (n_relations, n_entities, n_entities)
        self.R = nn.Parameter(torch.randn(n_relations, n_entities, n_entities))

    def forward(self, x):
        # x: (batch, n_entities, n_channels)
        # Produce the score f(s, r, o)
        scores = torch.einsum('bsc,rso,boc -> bsro', x, self.R, x)
        return scores

