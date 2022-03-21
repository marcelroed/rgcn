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
    pass


class LitRGCNEntityClassifier(pl.LightningModule):
    def __init__(self, n_nodes, n_relations, n_classes, channels: list[int]):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.n_classes = n_classes
        self.channels = channels
