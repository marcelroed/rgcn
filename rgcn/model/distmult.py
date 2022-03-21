from rgcn.layers.decoder import DistMultDecoder
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchkge.models.bilinear import DistMultModel

from rgcn.losses.sro import link_prediction_cross_entropy


class DistMult(nn.Module):
    def __init__(self, n_relations, n_entities, n_channels=50):
        super().__init__()
        self.decoder = DistMultDecoder(n_relations, n_channels)
        self.initializations = nn.Parameter(torch.randn(n_entities, n_channels, dtype=torch.float))
        self.n_channels = n_channels

    def forward(self, edge_index, edge_type):
        return self.decoder(self.initializations, edge_index, edge_type)

class LitDistMult(pl.LightningModule):
    def __init__(self, n_relations, n_entities, n_channels=50):
        super().__init__()
        self.n_relations = n_relations
        self.entities = n_entities
        self.n_channel = n_channels
        self.save_hyperparameters()
        self.model = torch.jit.script(DistMult(n_relations, n_entities, n_channels))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-1)

    def forward(self, edge_index, edge_type):
        scores = self.model(edge_index, edge_type)
        return scores

    def training_step(self, data):
        train_edge_index = data.edge_index[:, data.train_mask]
        train_edge_type = data.edge_type[data.train_mask]
        train_pos_index = data.train_pos_mask[data.train_mask]
        scores = self.model(train_edge_index, train_edge_type)
        loss = link_prediction_cross_entropy(scores, train_pos_index)
        self.log('train_loss', loss)
        return loss


class LitDistMultKGE(pl.LightningModule):
    def __init__(self, n_relations, n_entities, n_channels=50):
        super().__init__()
        self.n_relations = n_relations
        self.entities = n_entities
        self.n_channel = n_channels
        self.save_hyperparameters()
        self.model = DistMultModel(emb_dim=n_channels, n_relations=n_relations, n_entities=n_entities)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-1)

    def forward(self, edge_index, edge_type):
        scores = self.model.scoring_function(edge_index[0, :], edge_index[1, :], edge_type)
        return scores

    def training_step(self, data):
        train_edge_index = data.edge_index[:, data.train_mask]
        train_edge_type = data.edge_type[data.train_mask]
        train_pos_index = data.train_pos_mask[data.train_mask]
        scores = self.model.scoring_function(train_edge_index[0, :], train_edge_index[1, :], train_edge_type)
        loss = link_prediction_cross_entropy(scores, train_pos_index)
        self.log('train_loss', loss)
        return loss
