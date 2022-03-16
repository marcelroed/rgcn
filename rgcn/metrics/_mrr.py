from typing import Union, Literal

import torch
from dataclasses import dataclass


# def mean_reciprocal_rank(hrt, labels):
#     """
#     Mean reciprocal rank. For a graph neural network with true edges given by labels.
#     """
#     return torch.mean(1.0 / (torch.argsort(logits, dim=1)[:, :, ::-1] + 1).gather(1, labels.unsqueeze(1)))


@dataclass
class MRRResults:
    mrr: float
    hits_at_10: float
    hits_at_3: float
    hits_at_1: float

def mean_reciprocal_rank_and_hits(hrt_scores, test_edge_index, corrupt: Literal['head', 'tail']):
    assert corrupt in ['head', 'tail']

    # hrt_scores: (n_test_edges, n_nodes)
    perm = torch.argsort(hrt_scores, dim=1, descending=True)
    # Find the location of the true edges in the sorted list
    if corrupt == 'head':
        mask = perm == test_edge_index[0, :].view(-1, 1)
    else:
        mask = perm == test_edge_index[1, :].view(-1, 1)

    # Get the index of the true edges in the sorted list
    true_index = torch.nonzero(mask) + 1
    # Get the reciprocal rank of the true edges
    rr = 1.0 / true_index.float()

    # Get the mean reciprocal rank
    mrr = torch.mean(rr).item()

    # Get the hits@10 of the true edges
    hits10 = torch.sum(mask[:, :10], dim=1, dtype=torch.float).mean().item()
    # Get the hits@3 of the true edges
    hits3 = torch.sum(mask[:, :3], dim=1, dtype=torch.float).mean().item()
    # Get the hits@1 of the true edges
    hits1 = torch.sum(mask[:, :1], dim=1, dtype=torch.float).mean().item()
    return MRRResults(mrr, hits10, hits3, hits1)