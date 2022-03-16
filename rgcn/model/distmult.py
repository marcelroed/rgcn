from rgcn.layers.decoder import DistMultDecoder
import torch
import torch.nn as nn
import pytorch_lightning as pl

from rgcn.losses.sro import cross_entropy

class DistMult(nn.Module):
    def __init__(self, n_relations, n_entities, n_channels=50):
        super().__init__()
        self.decoder = DistMultDecoder(n_relations, n_channels)
        self.initializations = nn.Parameter(torch.randn(n_entities, n_channels))
        self.n_channels = n_channels

    def forward(self, sro_triples):
        return self.decoder(self.initializations, sro_triples)

class LitDistMult(pl.LightningModule):
    def __init__(self, n_relations, n_entities, n_channels=50):
        super().__init__()
        self.model = DistMult(n_relations, n_entities, n_channels)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, sro_triples):
        return self.model(sro_triples)

    def training_step(self, data):
        train_edge_index = data.edge_index[data.train_mask]
        train_edge_type = data.edge_type[data.train_mask]
        train_pos_index = data.train_pos_mask[data.train_mask]
        scores = self.model((train_edge_index[0], train_edge_type, train_edge_index[1]))
        return cross_entropy(scores, train_pos_index)
