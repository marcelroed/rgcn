from typing import Union, Literal

import torch
from dataclasses import dataclass
from tqdm.auto import trange


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
    print('Computing MRR')

    # hrt_scores: (n_test_edges, n_nodes)
    perm = torch.argsort(hrt_scores, dim=1, descending=True)
    # Find the location of the true edges in the sorted list
    if corrupt == 'head':
        mask = perm == test_edge_index[0, :].view(-1, 1)
    else:
        mask = perm == test_edge_index[1, :].view(-1, 1)

    # Get the index of the true edges in the sorted list
    true_index = torch.nonzero(mask)[:, 1] + 1
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


def _calc_mrr(emb, w, test_mask, triplets_to_filter, batch_size, filter=False):
    with torch.no_grad():
        test_triplets = triplets_to_filter[test_mask]
        s, r, o = test_triplets[:,0], test_triplets[:,1], test_triplets[:,2]
        test_size = len(s)

        if filter:
            metric_name = 'MRR (filtered)'
            triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter.tolist()}
            ranks_s = perturb_and_get_filtered_rank(emb, w, s, r, o, test_size,
                                                    triplets_to_filter, filter_o=False)
            ranks_o = perturb_and_get_filtered_rank(emb, w, s, r, o,
                                                    test_size, triplets_to_filter)
        else:
            metric_name = 'MRR (raw)'
            ranks_s = perturb_and_get_raw_rank(emb, w, o, r, s, test_size, batch_size)
            ranks_o = perturb_and_get_raw_rank(emb, w, s, r, o, test_size, batch_size)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed
        mrr = torch.mean(1.0 / ranks.float()).item()
        print("{}: {:.6f}".format(metric_name, mrr))

    return mrr

# Main evaluation function

def calc_mrr(emb, w, test_mask, triplets, batch_size=100, eval_p="filtered"):
    if eval_p == "filtered":
        mrr = _calc_mrr(emb, w, test_mask, triplets, batch_size, filter=True)
    else:
        mrr = _calc_mrr(emb, w, test_mask, triplets, batch_size)
    return mrr


def perturb_and_get_raw_rank(emb, w, a, r, b, test_size, batch_size=100):
    """ Perturb one element in the triplets"""
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    emb = emb.transpose(0, 1) # size D x V
    w = w.transpose(0, 1)     # size D x R
    for idx in trange(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = (idx + 1) * batch_size
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = emb[:,batch_a] * w[:,batch_r] # size D x E
        emb_ar = emb_ar.unsqueeze(2)           # size D x E x 1
        emb_c = emb.unsqueeze(1)               # size D x 1 x V

        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c)          # size D x E x V
        score = torch.sum(out_prod, dim=0).sigmoid() # size E x V
        target = b[batch_start: batch_end]

        _, indices = torch.sort(score, dim=1, descending=True)
        indices = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
        ranks.append(indices[:, 1].view(-1))
    return torch.cat(ranks)

# Utility functions for evaluations (filtered)

def filter(triplets_to_filter, target_s, target_r, target_o, num_nodes, filter_o=True):
    """Get candidate heads or tails to score"""
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)

    # Add the ground truth node first
    if filter_o:
        candidate_nodes = [target_o]
    else:
        candidate_nodes = [target_s]

    for e in range(num_nodes):
        triplet = (target_s, target_r, e) if filter_o else (e, target_r, target_o)
        # Do not consider a node if it leads to a real triplet
        if triplet not in triplets_to_filter:
            candidate_nodes.append(e)
    return torch.LongTensor(candidate_nodes)


def perturb_and_get_filtered_rank(emb, w, s, r, o, test_size, triplets_to_filter, filter_o=True):
    """Perturb subject or object in the triplets"""
    num_nodes = emb.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        candidate_nodes = filter(triplets_to_filter, target_s, target_r,
                                 target_o, num_nodes, filter_o=filter_o)
        if filter_o:
            emb_s = emb[target_s]
            emb_o = emb[candidate_nodes]
        else:
            emb_s = emb[candidate_nodes]
            emb_o = emb[target_o]
        target_idx = 0
        emb_r = w[target_r]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))

        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)


def mrr_with_dgl(x, w, test_edge_index, edge_type):
    triplets = torch.stack([test_edge_index[0], edge_type, test_edge_index[1]], dim=1)
    return calc_mrr(x, w, test_mask=torch.ones(*edge_type.shape, dtype=torch.bool), triplets=triplets, eval_p='unfiltered')