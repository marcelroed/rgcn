from __future__ import annotations
from typing import Literal

import torch
import pytorch_lightning as pl
from attr import define
from dbgpy import dbg
from pytorch_lightning.callbacks import EarlyStopping
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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
    loader = DataLoader([DATA])
    print(DATA)
    model = LitDistMult(NUM_RELATIONS, DATA.num_nodes, n_channels=2)
    wandb_logger = WandbLogger(name='rgcn_distmult', project='rgcn', save_dir='/tmp/wandb')
    # wandb_logger.watch(model, log_freq=10)
    trainer = pl.Trainer(max_epochs=100, callbacks=[EarlyStopping(monitor='train_loss')], gpus=0, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, loader)
    test(DATA.to(model.device), model)
