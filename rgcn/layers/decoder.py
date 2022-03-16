import torch
from torch import nn
from opt_einsum import contract


class DistMultDecoder(nn.Module):
    def __init__(self, n_relations, n_channels):
        super().__init__()
        # (n_relations, n_entities) expands to (n_relations, n_entities, n_entities)
        self.R_diagonal = nn.Parameter(torch.randn(n_relations, n_channels))

    def forward(self, x, sro_triples):
        # x: (batch, n_entities, n_channels)
        # Produce the score f(s, r, o)
        print('Running einsum')
        print(self.R_diagonal.shape)
        print(x.shape)
        scores = contract('sc,rc,oc -> sro', x, self.R_diagonal, x, backend='torch')
        print('Running einsum done')
        return scores


class ComplExDecoder(nn.Module):
    def __init__(self, n_relations, n_entities):
        super().__init__()
        # (n_relations, n_entities) expands to (n_relations, n_entities, n_entities)
        self.R_diagonal = nn.Parameter(torch.randn(n_relations, n_entities, dtype=torch.complex64))

    def forward(self, x):
        assert x.dtype == torch.complex64
        # x: (batch, n_entities, n_channels)
        # Produce the score f(s, r, o)
        scores = torch.einsum('bsc,ro,boc -> bsro', x.conj(), self.R_diagonal, x)
        return scores.real


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

