import pytorch_lightning
import torch.cuda
import wandb
from attrs import define, asdict
from pytorch_lightning.profiler import PyTorchProfiler
from torch_geometric.datasets import Entities
from torch_geometric.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from rgcn.data.datasets import EntityClassificationDataset

from rgcn.model.rgcn import LitRGCNEntityClassifier


@define
class ModelParameters:
    hidden_channels: list[int] = [16]
    l2_reg: float = 5e-4
    lr: float = 0.01
    epochs: int = 150

    n_basis_functions: int = 30
    decomposition_method: str = 'basis'

    def model_dict(self):
        d = asdict(self)
        d.pop('epochs')
        return d


def train_model(model_parameters: ModelParameters, dataset: EntityClassificationDataset):
    # Initialize model
    num_relations = dataset.edge_type.max().item() + 1
    num_classes = max(dataset.test_y.max().item(), dataset.train_y.max().item()) + 1
    model = LitRGCNEntityClassifier(dataset.num_nodes, num_relations, num_classes,
                                    **model_parameters.model_dict())

    # Initialize trainer
    wandb_logger = WandbLogger(name='rgcn_entity_classification', project='rgcn')
    # wandb_logger.watch(model, log_freq=10)
    trainer = Trainer(gpus=int(torch.cuda.is_available()), max_epochs=model_parameters.epochs,
                      callbacks=[],
                      # profiler=PyTorchProfiler(record_shapes=True, export_to_chrome=True)
                      log_every_n_steps=1,
                      logger=wandb_logger)
    dataloader = dataset.get_train_dataloader()
    val_dataloader = dataset.get_test_dataloader()
    trainer.fit(model, dataloader, val_dataloader)

    wandb_logger.finalize('success')

    return model, trainer


def evaluate_model(model: LitRGCNEntityClassifier, dataset: EntityClassificationDataset, trainer: Trainer):
    dataloader = dataset.get_test_dataloader()
    result = trainer.test(model, dataloader)
    print(result)


def main():
    dataset = EntityClassificationDataset.get_dataset('MUTAG')
    model, trainer = train_model(ModelParameters(), dataset)

    evaluate_model(model, dataset, trainer)


if __name__ == '__main__':
    main()
