import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import WordNet18, WordNet18RR

USE_RR = False
WordNet = WordNet18RR if USE_RR else WordNet18

from pytorch_lightning.loggers import WandbLogger
from rgcn.metrics import mean_reciprocal_rank_and_hits
from tqdm.auto import trange

from rgcn.model.distmult import LitDistMult

NUM_RELATIONS = 11 if USE_RR else 18
DATA = WordNet(f'data/wordnet18{"_rr" if USE_RR else ""}')[0]


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

    for i in range(0, num_relations):
        rel_indices = train_edge_type == i
        rel_edge_index = train_edge_index[:, rel_indices]
        rel_count = rel_indices.count_nonzero()
        rel_neg_edge_index = negative_sampling(edge_index=rel_edge_index,
                                               num_nodes=data.num_nodes,
                                               num_neg_samples=rel_count)
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
    return torch.stack((torch.ones(num_nodes, dtype=torch.long) * head, range_n), dim=0)


def get_tail_corrupted(head, tail, num_nodes):
    range_n = torch.arange(0, num_nodes, dtype=torch.long)
    return torch.stack((range_n, torch.ones(num_nodes, dtype=torch.long) * tail), dim=0)


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
            result.append(scores)
        return torch.stack(result)


def test(data, model):
    test_edge_index = data.edge_index[:, data.test_mask]
    test_edge_type = data.edge_type[data.test_mask]
    logits = generate_logits(test_edge_index, test_edge_type, data.num_nodes, model, get_head_corrupted)

    results = mean_reciprocal_rank_and_hits(logits, test_edge_index, corrupt='head')
    print(results)


if __name__ == '__main__':
    loader = DataLoader([knowledge_graph_negative_sampling(DATA, NUM_RELATIONS)])
    print(DATA)
    model = LitDistMult(NUM_RELATIONS, DATA.num_nodes, n_channels=50)
    # wandb_logger = WandbLogger(name='rgcn_distmult', project='rgcn')
    trainer = pl.Trainer(max_epochs=1000, callbacks=[EarlyStopping(monitor='train_loss')]) # , logger=wandb_logger)
    trainer.fit(model, loader)
    test(DATA, model)
