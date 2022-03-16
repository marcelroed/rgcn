from rgcn.metrics import mean_reciprocal_rank_and_hits
import torch


def test_mrr_and_hits():
    hrt_scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    hrt_scores = hrt_scores.expand(3, -1)
    for i in range(hrt_scores.shape[0]):
        hrt_scores[i] = hrt_scores[i][torch.randperm(hrt_scores.shape[1])]

    print(hrt_scores)
    edge_index = torch.tensor([[0, 1, 2, ], [1, 0, 3]])

    mrr, *hits = mean_reciprocal_rank_and_hits(hrt_scores, edge_index, corrupt='tail')
    print(mrr, hits)