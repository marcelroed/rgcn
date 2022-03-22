from __future__ import annotations

import os
from typing import Literal, Callable, List, Optional

import torch
from attr import define
from dbgpy import dbg
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import WordNet18, WordNet18RR, Entities
from rgcn.experiments.common import GraphData


# AIFB = Entities(f'data/aifb', 'AIFB')[0]
# MUTAG = Entities(f'data/mutag', 'MUTAG')[0]
# BGS = Entities(f'data/bgs', 'BGS')[0]
# AM = Entities(f'data/am', 'AM')[0]

class CustomDataset(InMemoryDataset):
    urls = {
        'FB15k-237': 'https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB-Toutanova',
        'FB15k': 'https://raw.githubusercontent.com/MichSchli/RelationPrediction/master/data/FB15k'
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data3.pt'

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'entities.dict', 'relations.dict', 'test.txt', 'train.txt',
            'valid.txt'
        ]

    def download(self):
        for file_name in self.raw_file_names:
            download_url(f'{self.urls[self.name]}/{file_name}', self.raw_dir)

    def process(self):
        with open(os.path.join(self.raw_dir, 'entities.dict'), 'r') as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            entities_dict = {key: int(value) for value, key in lines}

        with open(os.path.join(self.raw_dir, 'relations.dict'), 'r') as f:
            lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
            relations_dict = {key: int(value) for value, key in lines}

        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)

        size = {}
        for split in ['train', 'valid', 'test']:
            with open(os.path.join(self.raw_dir, f'{split}.txt'), 'r') as f:
                lines = [row.split('\t') for row in f.read().split('\n')[:-1]]
                src = [entities_dict[row[0]] for row in lines]
                rel = [relations_dict[row[1]] for row in lines]
                dst = [entities_dict[row[2]] for row in lines]
                edge_index = torch.cat((edge_index, torch.tensor([src, dst])), dim=-1)
                edge_type = torch.cat((edge_type, torch.tensor(rel)), dim=-1)
                size[split] = len(lines)

        print("here")
        data = Data(num_nodes=len(entities_dict),
                    edge_index=edge_index,
                    edge_type=edge_type,
                    train_mask=torch.cat((torch.ones(size['train']), torch.zeros(size['valid'] + size['test']))).bool(),
                    val_mask=torch.cat(
                        (torch.zeros(size['train']), torch.ones(size['valid']), torch.zeros(size['test']))).bool(),
                    test_mask=torch.cat((torch.zeros(size['train'] + size['valid']), torch.ones(size['test']))).bool())

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save((self.collate([data])), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


WORDNET18 = GraphData.from_dataset(WordNet18('data/wordnet18')[0], 'WN18')
FB15K_237 = GraphData.from_dataset(CustomDataset('data/fb15k_237', 'FB15k-237')[0], 'FB15k-237')


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
