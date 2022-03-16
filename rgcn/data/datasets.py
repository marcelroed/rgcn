import torch
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import WordNet18RR

from rgcn.model.distmult import LitDistMult

WN18RR_NUM_RELATIONS = 11
WN18RR_DATA = WordNet18RR('data')[0]


def knowledge_graph_negative_sampling(data, num_relations):
    # input: Data object with properties: edge_index, edge_type, train_mask, test_mask, val_mask, num_nodes
    # output: Data object with properties: above + train_pos_mask, train_neg_mask
    # Adds one negative training sample for each existing positive training sample
    neg_edge_index = torch.zeros(2, 0)
    neg_edge_type = torch.zeros(0)

    train_edge_type = data.edge_type[data.train_mask]
    train_edge_index = data.edge_index[:, data.train_mask]

    train_count = train_edge_type.shape[0]
    all_count = data.edge_type.shape[0]

    for i in range(0, num_relations):
        rel_indices = train_edge_type == i
        rel_edge_index = train_edge_index[:, rel_indices]
        rel_count = rel_indices.count_nonzero()
        rel_neg_edge_index = negative_sampling(edge_index=rel_edge_index,
                                               num_nodes=data.num_nodes,
                                               num_neg_samples=all_count)
        neg_edge_index = torch.cat((neg_edge_index, rel_neg_edge_index), dim=1)
        neg_edge_type = torch.cat((neg_edge_type, torch.ones(rel_count) * i))

    full_edge_index = torch.cat((data.edge_index, neg_edge_index), dim=-1)
    full_edge_type = torch.cat((data.edge_type, neg_edge_type), dim=-1)

    train_pos_mask = torch.cat((data.train_mask, torch.zeros(train_count)))
    train_neg_mask = torch.cat((torch.zeros(all_count), torch.ones(train_count)))
    train_mask = torch.cat((data.train_mask, torch.ones(train_count)))
    val_mask = torch.cat((data.val_mask, torch.zeros(train_count)))
    test_mask = torch.cat((data.test_mask, torch.zeros(train_count)))

    print('Returning data from knowledge_graph_negative_sampling')

    return Data(edge_index=full_edge_index,
                edge_type=full_edge_type,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                num_nodes=data.num_nodes,
                train_pos_mask=train_pos_mask,
                train_neg_mask=train_neg_mask)


if __name__ == '__main__':
    loader = DataLoader([knowledge_graph_negative_sampling(WN18RR_DATA, WN18RR_NUM_RELATIONS)])
    print(WN18RR_DATA)
    model = LitDistMult(WN18RR_NUM_RELATIONS, WN18RR_DATA.num_nodes)
    trainer = pl.Trainer()
    trainer.fit(model, loader)

    scores = model.forward()
