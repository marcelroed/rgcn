from typing import Union, List, Tuple

import torch_geometric
import torch
from torch_geometric.data import Data, DataLoader, InMemoryDataset


class SyntheticSmall(InMemoryDataset):
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'processed.pt'

    def download(self):
        pass

    def process(self):
        edge_index, edge_type = self._make_synthetic_graph()
        data, slices = self.collate([Data(edge_index=edge_index, edge_type=edge_type, train_mask=torch.ones_like(edge_type), test_mask=torch.ones_like(edge_type), val_mask=torch.ones_like(edge_type))])
        torch.save((data, slices), self.processed_paths[0])

    def __init__(self):
        super().__init__('data/synthetic_small')
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_mask = self.data.train_mask
        self.test_mask = self.data.test_mask
        self.val_mask = self.data.val_mask
        print(self.data)

    def _make_synthetic_graph(self):
        edge_index = torch.tensor([
            [0, 1, 2, 2, 2, 2, 2, 3, 4],
            [3, 0, 0, 1, 3, 3, 5, 4, 2],
        ], dtype=torch.long)
        edge_type = torch.tensor([0, 1, 2, 1, 0, 1, 3, 0, 2], dtype=torch.long)

        return edge_index, edge_type

        # # Create a random adjacency matrix
        # adj = torch.rand(n, n, dtype=torch.bool)
        # adj = adj.triu(diagonal=1)
        # adj = adj | adj.t()


if __name__ == '__main__':
    dataset = SyntheticSmall()
    data = dataset[0].edge_index
    print(data)
