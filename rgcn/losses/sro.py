import torch.nn.functional as F
import torch


def link_prediction_cross_entropy(sro_predicted, positive_mask, negative_per_positive=1, normalized=True):
    """
    Computes the cross entropy loss for the SRO model.
    """
    # sro_predicted: (num_edges)
    # positive_mask: (num_edges)
    w = negative_per_positive
    E = sro_predicted.shape[0]
    y = positive_mask.float()

    eps = 1e-15

    prob = torch.sigmoid(sro_predicted)
    # bce = F.binary_cross_entropy_with_logits(sro_predicted, positive_mask.float())
    bce = - torch.sum(y * torch.log(prob + eps) + (1 - y) * torch.log(1 - prob + eps))
    if normalized:
        bce /= (1 + w) * E
    return bce
