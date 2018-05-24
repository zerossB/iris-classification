import iris.functions as fn
from iris.neural import NeuralNetwork


def main():
    # dataset = fn.loadDataset()
    extract_data = fn.extractData()
    NN = NeuralNetwork(extract_data, 0.1)

    print(NN._model())


if __name__ == '__main__':
    main()
