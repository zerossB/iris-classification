from matplotlib import pyplot as plt
import numpy as np

dataset = []
ds_sl = []
ds_pl = []


def name2type(name):
    if name == 'Iris-setosa':
        return 0.0
    elif name == 'Iris-versicolor':
        return 0.5
    elif name == 'Iris-virginica':
        return 1.0


def loadDataset():
    with open("assets/iris.data", "r") as fp_data:
        for line in fp_data:
            try:
                sepal_length, sepal_width, petal_length, petal_width, name = line.split(
                    ',')
            except ValueError:
                pass
            else:
                dataset.append([
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width,
                    name2type(name.rstrip())
                ])
    return np.array(dataset)


def plot_by_name(dataset, x_name, y_name):
    plt.title(x_name + " X " + y_name)
    color = ['ro', 'bo', 'go']
    for name in dataset:
        plt.plot(dataset[name][x_name], dataset[name][y_name], color.pop(0))
    plt.show()


def plot_by_result(selected_dataset, neuralNetwork):
    plt.title("Resultado da Análise")

    color = ['ro', 'bo', 'go']
    for i in range(len(selected_dataset)):
        pred = neuralNetwork.prediction(
            selected_dataset[i][0], selected_dataset[i][1])
        if pred < 0.2:
            plt.plot(selected_dataset[i][0],
                     selected_dataset[i][1], color[0])
        elif pred > 0.8:
            plt.plot(selected_dataset[i][0],
                     selected_dataset[i][1], color[1])
        else:
            plt.plot(selected_dataset[i][0],
                     selected_dataset[i][1], color[2])
    plt.show()


def plotDataset(dataset, title):
    plt.title(title)
    plt.plot(dataset)
    plt.show()
