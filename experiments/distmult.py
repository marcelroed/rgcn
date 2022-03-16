import torch
from torch_geometric.datasets import WordNet18RR
from rgcn.model.distmult import LitDistMult
from torch_geometric.utils import train_test_split_edges, nega
import pytorch_lightning as pl
from pathlib import Path

DATA_DIRECTORY = Path('data/')

wn18pr = WordNet18RR(DATA_DIRECTORY / 'wn18rr')
data = train_test_split_edges(wn18pr[0]) # get positive and negative samples
model = LitDistMult()

trainer = pl.Trainer(gpus=0, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)



if __name__ == '__main__':
