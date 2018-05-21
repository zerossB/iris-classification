from matplotlib import pyplot as plt

dataset = {}
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
                if name.rstrip() in dataset:
                    dataset[name.rstrip()]['sepal_length'].append(
                        float(sepal_length))
                    dataset[name.rstrip()]['sepal_width'].append(
                        float(sepal_width))
                    dataset[name.rstrip()]['petal_length'].append(
                        float(petal_length))
                    dataset[name.rstrip()]['petal_width'].append(
                        float(petal_width))
                    dataset[name.rstrip()]['type'].append(
                        name2type(name.rstrip()))
                else:
                    dataset[name.rstrip()] = {
                        'sepal_length': [],
                        'sepal_width': [],
                        'petal_length': [],
                        'petal_width': [],
                        'type': []
                    }
    return dataset


def extract_data(dataset, x_name, y_name):
    ex_dataset = []
    for item in dataset:
        for index in range(len(dataset[item][x_name])):
            ex_dataset.append([dataset[item][x_name][index],
                               dataset[item][y_name][index],
                               dataset[item]['type'][index]])
    return ex_dataset


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
