from rgcn.model.distmult import LitDistMult
from rgcn.experiments.common import train_model, ModelConfig, GraphData, test_model
from rgcn.data.datasets import WORDNET18


def main():
    model = train_model(ModelConfig(
        model_class=LitDistMult, n_channels=100), WORDNET18, epochs=50)
    test_model(model, WORDNET18)


if __name__ == '__main__':
    main()
