from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from rgcn.layers.rgcn import RGCNConv, RGCNDirectConv
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
    def __init__(self, n_nodes, n_relations, layer_channels: list[int], decomposition_method, l2_reg, n_basis_functions):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.n_basis_functions = n_basis_functions
        self.l2_reg = l2_reg
        self.layers = nn.ModuleList([
            RGCNDirectConv(layer_channels[i], layer_channels[i + 1], self.n_relations, decomposition_method=decomposition_method, n_basis_functions=n_basis_functions)
            for i in range(len(layer_channels) - 1)
        ])

        # Precompute normalization constants
        self.register_buffer('normalization_constants', torch.zeros(self.n_relations, self.n_nodes, dtype=torch.long))

    def forward(self, x: OptTensor, edge_index, edge_type):
        [scatter(torch.ones_like(edge_index[1][edge_type == r]), edge_index[1][edge_type == r], out=self.normalization_constants[r]) for r in range(self.n_relations)]

        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer.forward(x, edge_index, edge_type, self.normalization_constants))
        last_layer = self.layers[-1]
        logits = last_layer.forward(x, edge_index, edge_type, self.normalization_constants)
        self.l2_norm()
        return logits

    def predict(self, x, edge_index, edge_type):
        logits = self.forward(x, edge_index, edge_type)
        return torch.softmax(logits, dim=1)

    def l2_norm(self):
        """Calculates the l2 norm of only the first layer's weights"""
        return self.layers[0].l2_norm()


class LitRGCNEntityClassifier(pl.LightningModule):
    def __init__(self, n_nodes, n_relations, n_classes,
                 hidden_channels: list[int], lr,
                 decomposition_method: Literal['none', 'basis', 'block'],
                 l2_reg: Optional[float], n_basis_functions: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.n_classes = n_classes
        self.hidden_channels = hidden_channels
        self.lr = lr
        self.l2_reg = l2_reg
        self.n_basis_functions = n_basis_functions
        self.model = torch.jit.script(RGCNEntityClassifier(n_nodes=self.n_nodes, n_relations=self.n_relations,
                                                           layer_channels=[self.n_nodes] + self.hidden_channels + [self.n_classes],
                                                           decomposition_method=decomposition_method, l2_reg=self.l2_reg, n_basis_functions=self.n_basis_functions))
        # self.model = RGCNEntityClassifier(self.n_nodes, self.n_relations, [self.n_nodes] + self.hidden_channels + [self.n_classes])

    def forward(self, x, edge_index, edge_type):
        return self.model(x, edge_index, edge_type)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Generate one_hot embeddings for all nodes
        one_hot_embeddings = None  # torch.eye(self.n_nodes)
        x = one_hot_embeddings

        edge_index = batch.edge_index
        edge_type = batch.edge_type

        logits = self(x, edge_index, edge_type)
        train_logits = logits[batch.train_idx]


        loss = F.cross_entropy(train_logits, batch.train_y)
        l2_loss = self.l2_reg * self.model.l2_norm()
        total_loss = loss + l2_loss

        self.log('train/loss', loss)
        self.log('train/l2_loss', l2_loss)
        self.log('train/total_loss', total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Generate one_hot embeddings for all nodes
        one_hot_embeddings = None  # torch.eye(self.n_nodes)
        x = one_hot_embeddings

        edge_index = batch.edge_index
        edge_type = batch.edge_type

        logits = self(x, edge_index, edge_type)
        test_logits = logits[batch.test_idx]

        # Calculate accuracy
        pred = torch.argmax(test_logits, dim=1)
        acc = torch.mean((pred == batch.test_y).float()).item()

        loss = F.cross_entropy(test_logits, batch.test_y)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_loss', loss)
        return {'val_acc': acc, 'val_loss': loss}

    def test_step(self, batch, batch_idx):
        # Generate one_hot embeddings for all nodes
        one_hot_embeddings = None  # torch.eye(self.n_nodes)
        x = one_hot_embeddings

        edge_index = batch.edge_index
        edge_type = batch.edge_type

        logits = self(x, edge_index, edge_type)
        test_logits = logits[batch.test_idx]

        # Calculate accuracy
        pred = torch.argmax(test_logits, dim=1)
        acc = torch.mean((pred == batch.test_y).float()).item()
        print(acc)
        return {'test_acc': acc}
