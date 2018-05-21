import iris.functions as fn
from iris.neural import NeuralNetwork


def main():
    x_label = 'petal_width'
    y_label = 'petal_length'

    dataset = fn.loadDataset()
    fn.plot_by_name(dataset, 'petal_width', 'petal_length')
    extract_dataset = fn.extract_data(dataset, x_label, y_label)

    NN = NeuralNetwork(extract_dataset, 0.1)
    error_log = []
    NN.l_rate = 0.3
    for i in range(1000):
        #error_log.append(NN.train_step()[0])
        error_log.append(NN.train_step_batch()[0])
    
    fn.plotDataset(error_log, "Log de Erros")

    fn.plot_by_result(extract_dataset, NN)


if __name__ == '__main__':
    main()
