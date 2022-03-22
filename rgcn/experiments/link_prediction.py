from rgcn.model.distmult import LitDistMult, LitComplEx
from rgcn.experiments.common import train_model, ModelConfig, GraphData, test_model
from rgcn.data.datasets import WORDNET18, FB15K_237


def main():
    model = train_model(ModelConfig(
        model_class=LitDistMult, n_channels=100), FB15K_237, epochs=200)
    test_model(model, FB15K_237)


if __name__ == '__main__':
    main()
