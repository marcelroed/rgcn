from rgcn.model.distmult import LitDistMult, LitDistMultKGE
from rgcn.experiments.common import train_model, ModelConfig, GraphData, test_model
from rgcn.data.datasets import DATA
from dbgpy import dbg


def compare_distmults():
    dbg(DATA)
    graph_data = GraphData.from_dataset(DATA)

    model = train_model(ModelConfig(
        model_class=LitDistMult, n_channels=100), graph_data, epochs=1000)
    model_2 = train_model(ModelConfig(
        model_class=LitDistMultKGE, n_channels=100), graph_data)


    test_model(model, graph_data)
    test_model(model_2, graph_data)


if __name__ == '__main__':
    compare_distmults()