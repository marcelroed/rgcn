from rgcn.layers.decoder import DistMultDecoder, ComplExDecoder
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torchkge.models.bilinear import DistMultModel
from tqdm.auto import trange

from rgcn.losses.sro import link_prediction_cross_entropy


class DistMult(nn.Module):
    def __init__(self, n_relations, n_entities, n_channels=50):
        super().__init__()
        self.decoder = DistMultDecoder(n_relations, n_channels)
        self.initializations = nn.Parameter(torch.randn(n_entities, n_channels, dtype=torch.float))
        #nn.init.xavier_uniform_(self.initializations.data)
        self.n_channels = n_channels

    def forward(self, edge_index, edge_type):
        return self.decoder(self.initializations, edge_index, edge_type)

class ComplEx(nn.Module):
    def __init__(self, n_relations, n_entities, n_channels=50):
        super().__init__()
        self.decoder = ComplExDecoder(n_relations, n_channels)
        self.initializations = nn.Parameter(torch.randn(n_entities, n_channels, dtype=torch.complex64))
        self.n_channels = n_channels

    def forward(self, edge_index, edge_type):
        return self.decoder(self.initializations, edge_index, edge_type)


def knowledge_graph_negative_sampling(data, num_relations):
    # input: Data object with properties: edge_index, edge_type, train_mask, test_mask, val_mask, num_nodes
    # output: Data object with properties: above + train_pos_mask, train_neg_mask
    # Adds one negative training sample for each existing positive training sample
    neg_edge_index = torch.zeros(2, 0, dtype=torch.long)
    neg_edge_type = torch.zeros(0, dtype=torch.long)

    train_edge_type = data.edge_type[data.train_mask]
    train_edge_index = data.edge_index[:, data.train_mask]

    train_count = train_edge_type.shape[0]
    all_count = data.edge_type.shape[0]

    #for i in trange(0, num_relations, desc='Generating negative samples'):
    for i in range(0, num_relations):
        rel_indices = train_edge_type == i
        rel_edge_index = train_edge_index[:, rel_indices]
        rel_count = rel_indices.long().sum()
        # print(i, rel_count)
        rel_neg_edge_index = negative_sampling(edge_index=rel_edge_index,
                                               num_nodes=data.num_nodes,
                                               num_neg_samples=rel_count).long()
        neg_edge_index = torch.cat((neg_edge_index, rel_neg_edge_index), dim=1)
        neg_edge_type = torch.cat((neg_edge_type, torch.ones(rel_count, dtype=torch.long) * i))

    full_edge_index = torch.cat((data.edge_index, neg_edge_index), dim=-1)
    full_edge_type = torch.cat((data.edge_type, neg_edge_type), dim=-1)

    train_pos_mask = torch.cat((data.train_mask, torch.zeros(train_count, dtype=torch.bool)))
    train_neg_mask = torch.cat((torch.zeros(all_count), torch.ones(train_count, dtype=torch.bool)))
    train_mask = torch.cat((data.train_mask, torch.ones(train_count, dtype=torch.bool)))
    val_mask = torch.cat((data.val_mask, torch.zeros(train_count, dtype=torch.bool)))
    test_mask = torch.cat((data.test_mask, torch.zeros(train_count, dtype=torch.bool)))

    #print('Returning data from knowledge_graph_negative_sampling')
    assert full_edge_index.dtype == torch.long, f'{full_edge_index.dtype=}'

    return Data(edge_index=full_edge_index,
                edge_type=full_edge_type,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                num_nodes=data.num_nodes,
                train_pos_mask=train_pos_mask,
                train_neg_mask=train_neg_mask)


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
        data = knowledge_graph_negative_sampling(data, self.n_relations)
        train_edge_index = data.edge_index[:, data.train_mask]
        train_edge_type = data.edge_type[data.train_mask]
        train_pos_index = data.train_pos_mask[data.train_mask]
        scores = self.model(train_edge_index, train_edge_type)
        loss = link_prediction_cross_entropy(scores, train_pos_index)
        self.log('train_loss', loss)
        return loss


class LitComplEx(pl.LightningModule):
    def __init__(self, n_relations, n_entities, n_channels=50):
        super().__init__()
        self.n_relations = n_relations
        self.entities = n_entities
        self.n_channel = n_channels
        self.save_hyperparameters()
        self.model = ComplEx(n_relations, n_entities, n_channels)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-1)

    def forward(self, edge_index, edge_type):
        scores = self.model(edge_index, edge_type)
        return scores

    def training_step(self, data):
        data = knowledge_graph_negative_sampling(data, self.n_relations)
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
        data = knowledge_graph_negative_sampling(data, self.n_relations)
        train_edge_index = data.edge_index[:, data.train_mask]
        train_edge_type = data.edge_type[data.train_mask]
        train_pos_index = data.train_pos_mask[data.train_mask]
        scores = self.model.scoring_function(train_edge_index[0, :], train_edge_index[1, :], train_edge_type)
        loss = link_prediction_cross_entropy(scores, train_pos_index)
        self.log('train_loss', loss)
        return loss
