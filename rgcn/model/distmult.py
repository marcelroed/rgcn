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

    def forward(self, edge_index, edge_type):
        return self.decoder(self.initializations, edge_index, edge_type)

class LitDistMult(pl.LightningModule):
    def __init__(self, n_relations, n_entities, n_channels=50):
        super().__init__()
        self.n_relations = n_relations
        self.entities = n_entities
        self.n_channel = n_channels
        self.save_hyperparameters()
        self.model = DistMult(n_relations, n_entities, n_channels)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, data):
        train_edge_index = data.edge_index[data.train_mask]
        train_edge_type = data.edge_type[data.train_mask]
        # train_pos_index = data.train_pos_mask[data.train_mask]
        scores = self.model(train_edge_index, train_edge_type)
        return scores

    def training_step(self, data):
        train_edge_index = data.edge_index[:, data.train_mask]
        train_edge_type = data.edge_type[data.train_mask]
        train_pos_index = data.train_pos_mask[data.train_mask]
        scores = self.model(train_edge_index, train_edge_type)
        loss = cross_entropy(scores, train_pos_index)
        self.log('train_loss', loss)
        return loss

def compute_mrr(edge_index, edge_type, model, test_edges):
    # Produce new edge_indices for corrupted test edges
    test_edge_index = torch.cat([edge_index[0][test_edges], edge_index[1][test_edges]], dim=0)
