import torch


# def mean_reciprocal_rank(hrt, labels):
#     """
#     Mean reciprocal rank. For a graph neural network with true edges given by labels.
#     """
#     return torch.mean(1.0 / (torch.argsort(logits, dim=1)[:, :, ::-1] + 1).gather(1, labels.unsqueeze(1)))


def mean_reciprocal_rank_and_hits(hrt_scores, test_edge_index):
    # hrt_scores: (n_test_edges, n_nodes)

    perm = torch.argsort(hrt_scores, dim=1, descending=True)


