import torch
import torch.nn as nn
import torch.nn.functional as F
from rgcn.layers.rgcn import RGCNConv
from rgcn.layers.decoder import DistMultDecoder
import pytorch_lightning as pl

class RGCNModel(nn.Module):
    def __init__(self, n_nodes, n_relations):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.initial_channels = 32
        self.hidden_channels = 64
        self.out_channels = 64
        self.conv1 = RGCNConv(self.initial_channels, self.hidden_channels, self.n_relations)
        self.conv2 = RGCNConv(self.hidden_channels, self.out_channels, self.n_relations)
        self.initial_embeddings = nn.Parameter(torch.randn(self.n_nodes, self.initial_channels))

        self.decoder = DistMultDecoder(self.n_relations, self.out_channels)

    def forward(self, edge_index, edge_type):
        x = self.initial_embeddings

        # Encoder
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))

        # Decoder
        score = self.decoder(x, edge_index, edge_type)

        return score


class RGCNEntityClassifier(nn.Module):
    def __init__(self, n_nodes, n_relations, layer_channels: list[int]):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.layers = nn.ModuleList([
            RGCNConv(layer_channels[i], layer_channels[i + 1], self.n_relations)
            for i in range(len(layer_channels) - 1)
        ])

    def forward(self, x, edge_index, edge_type):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index, edge_type))
        last_layer = self.layers[-1]
        logits = last_layer(x, edge_index, edge_type)
        return logits

    def predict(self, x, edge_index, edge_type):
        logits = self.forward(x, edge_index, edge_type)
        return torch.softmax(logits, dim=1)


class LitRGCNEntityClassifier(pl.LightningModule):
    def __init__(self, n_nodes, n_relations, n_classes, hidden_channels: list[int], lr):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.n_classes = n_classes
        self.hidden_channels = hidden_channels
        self.lr = lr
        self.model = RGCNEntityClassifier(self.n_nodes, self.n_relations, self.hidden_channels + [self.n_classes])

    def forward(self, x, edge_index, edge_type):
        return self.model(x, edge_index, edge_type)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, edge_index, edge_type, y = batch
        logits = self(x, edge_index, edge_type)
        loss = F.cross_entropy(logits, y)
        return {'loss': loss}
