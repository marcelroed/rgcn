from attrs import define
from torch_geometric.datasets import Entities
from torch_geometric.data import DataLoader
from pytorch_lightning import Trainer

from rgcn.model.rgcn import LitRGCNEntityClassifier


@define
class ModelParameters:
    hidden_dimensions: list[int] = []
    lr: float = 0.01
    epochs: int = 100

def train_model(model_parameters: ModelParameters, dataset: Entities):
    # Initialize model
    model = LitRGCNEntityClassifier(dataset.data.x.shape[0], dataset.num_relations, dataset.num_classes,
                                    [dataset.num_node_features, *model_parameters.hidden_dimensions, dataset.num_classes], lr=model_parameters.lr)


    # Initialize data loader
    data_loader = DataLoader(dataset, shuffle=True)

    # Initialize trainer
    trainer = Trainer(gpus=0, max_epochs=model_parameters.epochs)
    trainer.fit(model, dataset)

    return model


def evaluate_model(model, dataset):



if __name__ == '__main__':
    main()