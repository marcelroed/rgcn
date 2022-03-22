from rgcn.model.distmult import LitDistMult
from rgcn.experiments.common import train_model, ModelConfig, GraphData, test_model
from rgcn.data.datasets import DATA


def main():
    graph_data = GraphData.from_dataset(DATA)
    model = train_model(ModelConfig(
        model_class=LitDistMult, n_channels=100), graph_data, epochs=50)
    test_model(model, graph_data)


if __name__ == '__main__':
    main()
