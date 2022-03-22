from rgcn.model.distmult import LitDistMult, LitDistMultKGE
from rgcn.experiments.common import train_model, ModelConfig, GraphData, test_model
from rgcn.data.datasets import WORDNET18
from dbgpy import dbg


def compare_distmults():
    dbg(WORDNET18)
    model = train_model(ModelConfig(
        model_class=LitDistMult, n_channels=100), WORDNET18, epochs=1000)
    model_2 = train_model(ModelConfig(
        model_class=LitDistMultKGE, n_channels=100), WORDNET18)

    test_model(model, WORDNET18)
    test_model(model_2, WORDNET18)


if __name__ == '__main__':
    compare_distmults()
