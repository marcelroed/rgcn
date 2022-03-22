from __future__ import annotations
from attr import define
import torch
import pytorch_lightning as pl
from typing import Union, ClassVar, Optional, Literal, Any, Type

from dbgpy import dbg
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import trange

from rgcn.metrics import mean_reciprocal_rank_and_hits, mrr_with_dgl, average_mrr
from rgcn.model.distmult import LitDistMult, LitDistMultKGE


@define
class ModelConfig:
    n_channels: int
    model_class: Type

    # The rest of the model parameters are retrieved from the dataset.


@define
class GraphData:
    @define
    class TestFeatures:
        edge_index: torch.Tensor
        edge_type: torch.Tensor

    data_obj: InMemoryDataset
    n_relations: int
    n_entities: int
    name: str
    train_mask: torch.Tensor
    test_mask: torch.Tensor

    def get_train_loader(self) -> DataLoader:
        return DataLoader([self.data_obj])

    def get_test_features(self) -> GraphData.TestFeatures:
        return GraphData.TestFeatures(edge_index=self.data_obj.edge_index, edge_type=self.data_obj.edge_type)

    @classmethod
    def from_dataset(cls, dataset: InMemoryDataset, name="Data"):
        data_obj = dataset
        n_relations = torch.max(data_obj.edge_type).item() + 1
        n_entities = dataset.num_nodes
        train_mask = data_obj.train_mask if hasattr(data_obj, 'train_mask') else print(
            'Getting default train_mask') or torch.ones(data_obj.edge_index.shape[1], dtype=torch.bool)
        test_mask = data_obj.test_mask if hasattr(data_obj, 'test_mask') else print(
            'Getting default test_mask') or torch.ones(data_obj.edge_index.shape[1], dtype=torch.bool)
        return cls(data_obj=data_obj, n_relations=n_relations, n_entities=n_entities, train_mask=train_mask,
                   test_mask=test_mask, name=name)


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


def train_model(model_config: ModelConfig, dataset: GraphData, epochs=100, gpu=False) -> pl.LightningModule:
    lit_model: Union[LitDistMult, LitDistMultKGE] = model_config.model_class(n_relations=dataset.n_relations,
                                                                             n_entities=dataset.n_entities,
                                                                             n_channels=model_config.n_channels)
    loader = dataset.get_train_loader()

    wandb_logger = WandbLogger(name=f'{lit_model.__class__.__name__.lower()}-{dataset.name}',
                               project='rgcn', save_dir='/tmp/wandb')
    # wandb_logger.watch(model, log_freq=10)

    # trainer = pl.Trainer(max_epochs=epochs, callbacks=[EarlyStopping(monitor='train_loss')], gpus=int(gpu), logger=wandb_logger, log_every_n_steps=1)
    trainer = pl.Trainer(max_epochs=epochs, gpus=int(gpu), logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(lit_model, loader)
    wandb_logger.close()

    return lit_model


def test_model(model: pl.LightningModule, dataset: GraphData):
    model.eval()
    print(f'Testing model {model.__class__.__name__}')
    test_features = dataset.get_test_features()
    edge_index, edge_type = test_features.edge_index, test_features.edge_type

    test_edge_index = edge_index[:, dataset.test_mask]
    test_edge_type = edge_type[dataset.test_mask]

    logits = generate_logits(test_edge_index, test_edge_type, dataset.n_entities, model, get_head_corrupted)
    results_head = mean_reciprocal_rank_and_hits(logits, test_edge_index.to(logits), corrupt='head')
    print(results_head)

    logits = generate_logits(test_edge_index, test_edge_type, dataset.n_entities, model, get_tail_corrupted)
    results_tail = mean_reciprocal_rank_and_hits(logits, test_edge_index.to(logits), corrupt='tail')
    print(results_tail)

    results = average_mrr(results_head, results_tail)
    print(results)

    # dgl_mrr = mrr_with_dgl(model.model.initializations, model.model.decoder.R_diagonal, test_edge_index, test_edge_type)
    # dbg(dgl_mrr)
