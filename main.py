import iris.functions as fn
from iris.neural import NeuralNetwork


def main():
    dataset = fn.loadDataset()
    NN = NeuralNetwork(dataset, 0.1)
    
    


if __name__ == '__main__':
    main()
