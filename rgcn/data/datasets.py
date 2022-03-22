from __future__ import annotations
from typing import Literal

import torch
import pytorch_lightning as pl
from attr import define
from dbgpy import dbg
from pytorch_lightning.callbacks import EarlyStopping
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import WordNet18, WordNet18RR, Entities

from rgcn.data.synthetic import SyntheticSmall

USE_RR = False
WordNet = WordNet18RR if USE_RR else WordNet18

from pytorch_lightning.loggers import WandbLogger
from rgcn.metrics import mean_reciprocal_rank_and_hits, mrr_with_dgl
from tqdm.auto import trange

from rgcn.model.distmult import LitDistMult, LitDistMultKGE

NUM_RELATIONS = 11 if USE_RR else 18
DATA = WordNet(f'data/wordnet18{"_rr" if USE_RR else ""}')[0]

# AIFB = Entities(f'data/aifb', 'AIFB')[0]
# MUTAG = Entities(f'data/mutag', 'MUTAG')[0]
# BGS = Entities(f'data/bgs', 'BGS')[0]
# AM = Entities(f'data/am', 'AM')[0]

@define
class EntityClassificationDataset:
    data: Entities

    @classmethod
    def get_dataset(cls, name: Literal['AIFB', 'MUTAG', 'BGS', 'AM']) -> EntityClassificationDataset:
        data_object = Entities(f'data/{name}', name)[0]
        return cls(data_object[0])

@define
class DatasetDescription:
    dataset: Data

    @classmethod
    def from_entity_dataset(cls, entity_dataset: str):
        cls(dataset=Entities(f'data/{entity_dataset.lower()}', entity_dataset.upper())[0])


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

    for i in trange(0, num_relations, desc='Generating negative samples'):
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

    print('Returning data from knowledge_graph_negative_sampling')
    assert full_edge_index.dtype == torch.long, f'{full_edge_index.dtype=}'

    return Data(edge_index=full_edge_index,
                edge_type=full_edge_type,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                num_nodes=data.num_nodes,
                train_pos_mask=train_pos_mask,
                train_neg_mask=train_neg_mask)


def get_head_corrupted(head, tail, num_nodes):
    range_n = torch.arange(0, num_nodes, dtype=torch.long)
    return torch.stack((range_n, torch.ones(num_nodes, dtype=torch.long) * tail), dim=0)


def get_tail_corrupted(head, tail, num_nodes):
    range_n = torch.arange(0, num_nodes, dtype=torch.long)
    return torch.stack((torch.ones(num_nodes, dtype=torch.long) * head, range_n), dim=0)


def generate_logits(test_edge_index, test_edge_type, num_nodes, model, corruption_func):
    test_count = test_edge_type.shape[0]
    result = []
    model.eval()
    with torch.no_grad():
        for i in trange(0, test_count, desc='Generating scores'):
            head, tail = test_edge_index[:, i].tolist()
            corrupted_edge_type = test_edge_type[i].repeat(num_nodes)
            corrupted_edge_index = corruption_func(head, tail, num_nodes)
            scores = model.forward(corrupted_edge_index, corrupted_edge_type)
            result.append(scores.detach())
        return torch.stack(result)


def test(data, model: LitDistMult):
    test_edge_index = data.edge_index[:, data.test_mask]
    test_edge_type = data.edge_type[data.test_mask]
    logits = generate_logits(test_edge_index, test_edge_type, data.num_nodes, model, get_head_corrupted)
    results_head = mean_reciprocal_rank_and_hits(logits, test_edge_index.to(logits), corrupt='head')
    logits = generate_logits(test_edge_index, test_edge_type, data.num_nodes, model, get_tail_corrupted)
    results_tail = mean_reciprocal_rank_and_hits(logits, test_edge_index.to(logits), corrupt='tail')
    results = (results_head.mrr + results_tail.mrr) / 2
    dbg(results)

    dgl_mrr = mrr_with_dgl(model.model.initializations, model.model.decoder.R_diagonal, test_edge_index, test_edge_type)
    dbg(dgl_mrr)



if __name__ == '__main__':
    loader = DataLoader([knowledge_graph_negative_sampling(DATA, NUM_RELATIONS)])
    print(DATA)
    model = LitDistMult(NUM_RELATIONS, DATA.num_nodes, n_channels=2)
    wandb_logger = WandbLogger(name='rgcn_distmult', project='rgcn', save_dir='/tmp/wandb')
    # wandb_logger.watch(model, log_freq=10)
    trainer = pl.Trainer(max_epochs=100, callbacks=[EarlyStopping(monitor='train_loss')], gpus=0, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, loader)
    test(DATA.to(model.device), model)
