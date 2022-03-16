from rgcn.metrics import mean_reciprocal_rank_and_hits
import torch


def test_mrr_and_hits():
    hrt_scores = torch.tensor([
        [.9, .7, .4],
        [.6, .5, .4],
        [.4, .6, .5],
        [.2, .4, .5]
    ])

    print(hrt_scores)
    edge_index = torch.tensor([[0, 1, 1, 0],
                               [1, 2, 0, 1]])

    results = mean_reciprocal_rank_and_hits(hrt_scores, edge_index, corrupt='head')
    print(results)