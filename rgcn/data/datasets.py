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

from rgcn.experiments.common import GraphData

from rgcn.model.distmult import LitDistMult, LitDistMultKGE

WORDNET18 = GraphData.from_dataset(WordNet18('data/wordnet18')[0], 'WN18')


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
