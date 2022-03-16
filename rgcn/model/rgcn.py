import torch
import torch.nn as nn
import torch.nn.functional as F
from rgcn.layers.rgcn import RGCNConv

class RGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels)

    def forward(self, x, edge_index):
