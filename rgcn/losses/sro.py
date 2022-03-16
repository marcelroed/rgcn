import torch.nn.functional as F


def cross_entropy(sro_predicted, positive_mask, negative_per_positive=1):
    """
    Computes the cross entropy loss for the SRO model.
    """
    # sro_predicted: (num_edges)
    # positive_mask: (num_edges)
    w = negative_per_positive
    E = sro_predicted.shape[0]
    return - 1 / ((1 + w) * E) * F.binary_cross_entropy_with_logits(sro_predicted, positive_mask).sum()
